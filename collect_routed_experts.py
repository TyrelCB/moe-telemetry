#!/usr/bin/env python3
"""Calibration-trace collector for MoE routed-expert telemetry (vLLM >=0.28).

Sends a calibration prompt set to a vLLM server started with
--enable-return-routed-experts, requests per-token routed expert IDs,
decodes the base64 .npy traces, and accumulates a (layers x experts)
usage-frequency matrix plus per-sample traces.

Usage:
  collect_routed_experts.py --base-url http://localhost:8123 --dataset prompts.jsonl --out runs/run1
  collect_routed_experts.py --serve <model-path> ...   (launches its own server, tears it down after)

Dataset: JSONL, one per line: {"messages": [...]} or {"prompt": "text"}.
Built-in default set (code/english/reasoning mix) used if --dataset omitted.
"""
import argparse, base64, io, json, os, signal, subprocess, sys, time, urllib.request, urllib.error
import numpy as np

DEFAULT_PROMPTS = [
    "def fib(n):\n    a, b = 0, 1",
    "Explain why the sky is blue in one sentence.",
    "Translate to French: The book is on the table.",
    "What is 17 * 23? Show your work.",
    "Write a haiku about winter.",
    "The capital of Australia is",
    "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name HAVING COUNT(o.id) > 5;",
    "If all roses are flowers and some flowers fade quickly, can we conclude that all roses fade quickly? Explain.",
    "class Stack:\n    def __init__(self):\n        self.items = []",
    "Summarize the plot of Romeo and Juliet in three bullet points.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr",
    "A train leaves Denver at 60 mph. Another leaves simultaneously at 75 mph in the opposite direction. How far apart after 3 hours?",
    "What is the time complexity of merge sort and why?",
    "Écrire une phrase en anglais utilisant le mot 'malheureusement'.",
    "Given a binary tree, write Python to compute its maximum depth.",
    "Compare TCP and UDP in a table.",
    "Riff: If it's not about the money, what is business really about?",
    "PyTorch: how do I move a tensor to GPU and detach gradients?",
    "The Pythagorean theorem states that",
    "List three risks of large language models in healthcare and one mitigation each.",
    "for i in range(10):\n    print(i * i, end=' ')",
    "Balance this equation: Fe + O2 -> Fe2O3",
    "Write a JSON object describing a book with title, author, and year.",
    "Why do we need synchronization primitives in multithreaded programs?",
    "Rewrite politely: 'Your invoice is late. Pay now.'",
    "Explain gradient descent to a bright 12-year-old.",
    "int x = 5; x += 3; x *= 2; what is x?",
    "Describe the water cycle in five steps.",
    "Convert 255 to hexadecimal and binary.",
    "What does 'return' do inside a Python generator?",
]

def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def wait_healthy(base, proc=None, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            sys.exit("server process died; see server log")
        try:
            with urllib.request.urlopen(base + "/health", timeout=5):
                return
        except Exception:
            time.sleep(5)
    sys.exit("server never became healthy")

def load_dataset(path):
    items = []
    if path:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                items.append(d if "messages" in d else {"messages": [{"role": "user", "content": d["prompt"]}]})
    else:
        items = [{"messages": [{"role": "user", "content": p}]} for p in DEFAULT_PROMPTS]
    return items

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8123")
    ap.add_argument("--model", default=None, help="model id (default: first from /v1/models)")
    ap.add_argument("--dataset", default=None, help="JSONL of {messages:[...]} or {prompt:...}")
    ap.add_argument("--out", required=True, help="output run directory")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--save-raw", action="store_true", help="save per-sample routed_experts arrays")
    ap.add_argument("--serve", default=None, help="model path: launch our own server (tears it down after)")
    ap.add_argument("--serve-extra", default="", help="extra args for --serve, single string")
    args = ap.parse_args()

    proc = None
    if args.serve:
        port = int(args.base_url.rsplit(":", 1)[-1].rstrip("/"))
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", args.serve, "--enable-return-routed-experts",
               "--max-model-len", "4096", "--gpu-memory-utilization", "0.66",
               "--port", str(port)] + args.serve_extra.split()
        log = open(os.path.join(os.path.dirname(os.path.abspath(args.out)) or ".", "server.log"), "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        print(f"launched server pid {proc.pid}, log: {log.name}")
    base = args.base_url.rstrip("/")
    wait_healthy(base, proc)

    model = args.model
    if not model:
        model = json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"]

    items = load_dataset(args.dataset)
    os.makedirs(args.out, exist_ok=True)
    counts = None          # (L, E) int64 usage counts
    sample_meta = []
    t0 = time.time()
    for i, item in enumerate(items):
        payload = {"model": model, "messages": item["messages"],
                   "max_tokens": args.max_tokens, "temperature": args.temperature,
                   "return_routed_experts": True}
        r = post(base + "/v1/chat/completions", payload)
        choice = r["choices"][0]
        if not choice.get("routed_experts"):
            print(f"sample {i}: NO TRACE (backend returned none) — is server started with --enable-return-routed-experts?", file=sys.stderr)
            continue
        arr = np.load(io.BytesIO(base64.b64decode(choice["routed_experts"])))  # (tokens, L, topk)
        L, K = arr.shape[1], arr.shape[2]
        E = int(arr.max()) + 1
        if counts is None:
            counts = np.zeros((L, E), dtype=np.int64)
        flat = arr.reshape(-1, L, K)  # (tokens, L, topk)
        for l in range(L):
            np.add.at(counts[l], flat[:, l, :], 1)
        if args.save_raw:
            os.makedirs(os.path.join(args.out, "traces"), exist_ok=True)
            np.save(os.path.join(args.out, "traces", f"{i:05d}.npy"), arr)
        sample_meta.append({"idx": i, "tokens": arr.shape[0], "layers": L, "topk": K,
                            "prompt_tokens": r["usage"]["prompt_tokens"],
                            "completion_tokens": r["usage"]["completion_tokens"]})
        print(f"[{i+1}/{len(items)}] tokens={arr.shape[0]} shape={arr.shape} elapsed={time.time()-t0:.0f}s")

    if counts is None:
        sys.exit("no traces collected")
    total = counts.sum()
    frac = counts / max(total, 1)
    # dead / rare experts per layer (expected uniform = 1/E of traffic)
    uniform = 1.0 / counts.shape[1]
    dead = [(int(l), int(e)) for l in range(counts.shape[0]) for e in range(counts.shape[1]) if counts[l, e] == 0]
    rare = [(int(l), int(e), float(counts[l, e] / max(counts[l].sum(), 1)))
            for l in range(counts.shape[0]) for e in range(counts.shape[1])
            if 0 < counts[l, e] < 0.1 * uniform * counts[l].sum()]
    summary = {
        "model": model, "samples": len(sample_meta),
        "total_tokens": sum(m["tokens"] for m in sample_meta),
        "elapsed_s": round(time.time() - t0, 1),
        "layers": int(counts.shape[0]), "experts_per_layer": int(counts.shape[1]),
        "topk": sample_meta[0]["topk"] if sample_meta else None,
        "dead_experts": dead, "rare_experts_lt_10pct_uniform": rare[:200],
        "load_gini_mean": None,
    }
    try:  # gini per layer as a load-balance measure
        gs = []
        for l in range(counts.shape[0]):
            x = np.sort(counts[l].astype(float)); n = len(x)
            gs.append(float(((2 * np.arange(1, n + 1) - n - 1) * x).sum() / (n * x.sum())) if x.sum() else 0.0)
        summary["load_gini_mean"] = float(np.mean(gs))
        summary["load_gini_per_layer"] = gs
    except Exception:
        pass
    np.save(os.path.join(args.out, "counts.npy"), counts)
    with open(os.path.join(args.out, "samples.jsonl"), "w") as f:
        for m in sample_meta:
            f.write(json.dumps(m) + "\n")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    manifest = {"model": model, "base_url": base, "dataset": args.dataset,
                "max_tokens": args.max_tokens, "temperature": args.temperature,
                "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "argv": sys.argv[1:]}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {args.out}/counts.npy (L={counts.shape[0]}, E={counts.shape[1]}) "
          f"summary.json samples.jsonl — {len(dead)} dead experts, gini={summary['load_gini_mean']}")
    if proc is not None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=60)

if __name__ == "__main__":
    main()
