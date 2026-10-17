#!/usr/bin/env python3
"""pass@1 code eval: HumanEval-style prompts -> generate -> execute tests.

Chat-endpoint generation with thinking disabled; accepts either a restated
full function or a raw continuation; strips leaked prose before the first
'def'. Appends the official HumanEval test and calls check(entry_point).

Usage: pass_at_1.py --url http://host:PORT [--model M] [--n 24] [--out FILE]
"""
import argparse, json, re, subprocess, sys, tempfile, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--model", default=None)
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--out", required=True)
args = ap.parse_args()

probs = []
for line in open("humaneval_local.jsonl"):
    if len(probs) >= args.n:
        break
    probs.append(json.loads(line))

def gen(prompt):
    payload = {"model": args.model,
               "messages": [{"role": "system",
                             "content": "You complete Python functions. Output only the code, no explanations, no markdown fences."},
                            {"role": "user", "content": prompt}],
               "max_tokens": 512, "temperature": 0.0,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    t = r["choices"][0]["message"]["content"]
    return re.sub(r"^```[a-zA-Z]*\n|\n```\s*$", "", t.strip())

def assemble(prompt, out, entry):
    code = out if f"def {entry}" in out else prompt + out
    keep, seen_def = [], False
    for ln in code.split("\n"):
        if ln.startswith("def "):
            seen_def = True
        if seen_def and ln and not ln[0].isspace() and not ln.startswith("def "):
            break  # first top-level statement after the function ends it
        keep.append(ln)
    return "\n".join(keep)

results = []
for i, p in enumerate(probs):
    out = gen(p["prompt"])
    if "def " in out and f"def {p['entry_point']}" not in out:
        out = out[out.find("def "):]  # strip leaked thinking prose before code
    code = assemble(p["prompt"], out, p["entry_point"])
    code += "\n" + p["test"] + "\ncheck(" + p["entry_point"] + ")\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(code)
        path = tf.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=20)
        ok = r.returncode == 0
    except Exception:
        ok = False
    results.append({"idx": p["task_id"], "pass": ok})
    print(f"[{i+1}/{len(probs)}] {p['task_id']}: {'PASS' if ok else 'FAIL'}", flush=True)

npass = sum(1 for r in results if r["pass"])
json.dump({"pass_at_1": npass / len(results), "solved": npass,
           "n": len(results), "results": results},
          open(args.out, "w"), indent=2)
print(f"pass@1 = {npass}/{len(results)} -> {args.out}")
