#!/usr/bin/env python3
"""Compare routing counts of baseline vs pruned replay (after index remap).

Usage: compare_pruned.py BASE_COUNTS PRUNED_COUNTS PRUNE_MANIFEST
"""
import json, sys
import numpy as np

base = np.load(sys.argv[1])                       # (L, 256)
pruned = np.load(sys.argv[2])                     # (L, N)
man = json.load(open(sys.argv[3]))
E, N = man["E"], man["N"]
drop = {int(k): set(v) for k, v in man["drop"].items()}

L = base.shape[0]
assert pruned.shape == (L, N), (pruned.shape, L, N)

tot_base_kept = 0.0
tot_pruned = 0.0
newly_dead = []
gini_b, gini_p = [], []
diffs = []
tvs = []

def gini(v):
    v = np.sort(v.astype(np.float64))
    n = len(v)
    if v.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))

for l in range(L):
    keep = [e for e in range(E) if e not in drop[l]]
    b = base[l, keep]
    p = pruned[l]
    tot_base_kept += b.sum()
    tot_pruned += p.sum()
    newly_dead += [(l, keep[i], int(p[i])) for i in range(N) if p[i] == 0 and b[i] > 0]
    gini_b.append(gini(b))
    gini_p.append(gini(p))
    # compare NORMALIZED per-layer distributions (sample-size invariant);
    # only experts with >=0.1% baseline mass in this layer
    if b.sum() > 0 and p.sum() > 0:
        bn, pn = b / b.sum(), p / p.sum()
        m = bn >= 0.001
        if m.any():
            diffs.append(np.abs(pn[m] - bn[m]))   # absolute prob mass drift
        tvs.append(0.5 * float(np.abs(bn - pn).sum()))

d = np.concatenate(diffs)
print(json.dumps({
    "tokens_routed_baseline_kept_experts": int(tot_base_kept),
    "tokens_routed_pruned": int(tot_pruned),
    "gini_baseline_kept_mean": round(float(np.mean(gini_b)), 4),
    "gini_pruned_mean": round(float(np.mean(gini_p)), 4),
    "mass_drift_mean": round(float(d.mean()), 6),
    "mass_drift_p95": round(float(np.percentile(d, 95)), 5),
    "mass_drift_max": round(float(d.max()), 5),
    "layer_tv_mean": round(float(np.mean(tvs)), 5),
    "layer_tv_max": round(float(np.max(tvs)), 5),
    "newly_dead_count": len(newly_dead),
    "newly_dead_after_prune": newly_dead[:20],
}, indent=2))
