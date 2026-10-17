#!/usr/bin/env python3
"""Teacher-forced logprob probe: dump mean prompt_logprob/token per sample.

Run once per server (sequential on one GPU), then diff the two dumps.

Usage: logprob_equivalence.py DATASET.jsonl --url http://host:PORT --out dump.json [--n 24]
Then:  logprob_equivalence.py --diff base_dump.json pruned_dump.json
"""
import argparse, json, statistics, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("dataset", nargs="?")
ap.add_argument("--url")
ap.add_argument("--out")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--diff", nargs=2)
args = ap.parse_args()

if args.diff:
    a = json.load(open(args.diff[0])); b = json.load(open(args.diff[1]))
    dd = []
    for k in a:
        if k in b:
            dd.append(b[k] - a[k])
    print(json.dumps({
        "n": len(dd),
        "baseline_mean": round(statistics.mean(a.values()), 4),
        "pruned_mean": round(statistics.mean(b.values()), 4),
        "mean_delta": round(statistics.mean(dd), 5),
        "max_worsening": round(max(dd), 5),
        "max_improvement": round(min(dd), 5),
        "n_better_or_equal": sum(1 for d in dd if d >= 0),
    }, indent=2))
    raise SystemExit

samples = [json.loads(l) for l in open(args.dataset)][: args.n]
res = {}
for i, s in enumerate(samples):
    prompt = s.get("prompt") or "\n".join(m["content"] for m in s["messages"])
    payload = {"prompt": prompt, "max_tokens": 1, "temperature": 0,
               "prompt_logprobs": 1}
    req = urllib.request.Request(args.url.rstrip("/") + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    pl = r["choices"][0].get("prompt_logprobs") or r.get("prompt_logprobs")
    if not pl:
        print(f"[{i}] no prompt_logprobs in response"); continue
    lps = [step[next(iter(step))]["logprob"] for step in pl if step]
    m = sum(lps) / len(lps)
    res[str(i)] = m
    print(f"[{i+1}/{len(samples)}] mean_lp={m:.4f}", flush=True)

json.dump(res, open(args.out, "w"))
print("wrote", args.out)
