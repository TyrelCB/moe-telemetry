#!/usr/bin/env python3
"""Build guided drop-set from chunk-ablation scores: per layer rank the
bottom-pool chunks by measured dlogprob (ascending), drop the 4 cheapest.

Usage: make_guided_drop.py SCORES_JSON GUIDED_CANDS_JSONL -> guided_drop.json
"""
import json, sys

sc = json.load(open(sys.argv[1]))["results"]
by_unit = {(r["layer"], r["chunk"]): r["mean_dlogprob_token"] for r in sc}
cands = {}
for line in open(sys.argv[2]):
    r = json.loads(line)
    cands.setdefault((r["layer"], r["chunk"]), []).append(r["expert"])

nchunks = 4
layers = sorted({l for l, _ in cands})
drop = {}
for l in layers:
    cs = sorted({c for ll, c in cands if ll == l},
                key=lambda c: by_unit.get((l, c), 0.0))
    drop[l] = sorted(e for c in cs[:nchunks] for e in cands[(l, c)])
json.dump({str(l): drop[l] for l in drop}, open("guided_drop.json", "w"))
sizes = sorted({len(v) for v in drop.values()})
print("layers:", len(drop), "| drop-set sizes:", sizes)
