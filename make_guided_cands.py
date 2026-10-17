#!/usr/bin/env python3
"""Guided-scoring candidates: per layer, the POOL lowest-count experts, split
into 8-expert chunks by count rank. Chunks are the scoring unit; the drop-set
is later rebuilt from measured chunk importance, not count rank.

Usage: make_guided_cands.py COUNTS_NPY POOL=96  -> guided_cands.jsonl
"""
import json, sys
import numpy as np

counts = np.load(sys.argv[1])
POOL = int(sys.argv[2]) if len(sys.argv) > 2 else 96
n = 0
with open("guided_cands.jsonl", "w") as f:
    for l in range(counts.shape[0]):
        order = np.argsort(counts[l])                 # ascending count
        pool = order[:POOL]
        for chunk in range(POOL // 8):
            for e in pool[chunk * 8:(chunk + 1) * 8]:
                f.write(json.dumps({"layer": l, "expert": int(e), "chunk": chunk}) + "\n")
                n += 1
print(f"wrote guided_cands.jsonl: {n} candidates "
      f"({counts.shape[0]} layers x {POOL // 8} chunks x 8)")
