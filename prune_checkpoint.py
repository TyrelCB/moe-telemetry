#!/usr/bin/env python3
"""Materialize a pruned MoE checkpoint from calibration counts.

Expert-pruning analogue of head/width pruning (see ChatGPT thread): drop the
experts that showed zero routing on real workload, plus pad each layer to a
uniform N_keep (HF requires a single num_experts across layers) by dropping
the lowest-count experts. Router rows are pruned + compacted in the same pass.

Usage: prune_checkpoint.py SRC_DIR COUNTS_NPY DST_DIR [--nkeep 241] [--mtp-drop-from 0]
"""
import argparse, json, os, re, shutil, sys
import numpy as np
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("counts"); ap.add_argument("dst")
ap.add_argument("--nkeep", type=int, default=None,
                help="experts kept per layer (default: min kept when dropping only count==0)")
ap.add_argument("--drop-file", default=None,
                help="JSON {layer: [experts...]} explicit drop sets (overrides --nkeep counts)")
ap.add_argument("--mtp-drop-from", type=int, default=0,
                help="main-MoE layer whose drop-set is applied to the MTP layer's experts")
args = ap.parse_args()

counts = np.load(args.counts)                 # (L, E)
L, E = counts.shape
if args.drop_file:
    df = json.load(open(args.drop_file))
    drop = {l: sorted(set(int(e) for e in df[str(l)])) for l in range(L)}
    N = min(E - len(drop[l]) for l in range(L))
    assert all(E - len(d) == N for d in drop.values()), "uniform drop count required"
else:
    N = args.nkeep or int((counts > 0).sum(1).min())
    drop = {}
    for l in range(L):
        order = np.argsort(counts[l])             # ascending count
        drop[l] = sorted(int(e) for e in order[:E - N])
keep = {l: [e for e in range(E) if e not in set(drop[l])] for l in range(L)}
remap = {l: {old: new for new, old in enumerate(keep[l])} for l in range(L)}
print(f"layers={L} E={E} -> keep {N}/layer; drops per layer: "
      + str({l: len(drop[l]) for l in range(L)}))

mtp_drop = set(drop[args.mtp_drop_from])
mtp_keep = [e for e in range(E) if e not in mtp_drop]

os.makedirs(args.dst, exist_ok=True)
src = args.src
idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
wm = idx["weight_map"]

expert_re = re.compile(r"^(?:model\.language_model|mtp)\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")
mtp_prefix = "mtp.layers."
gate_re = re.compile(r"^(?:(model\.language_model)|mtp)\.layers\.(\d+)\.mlp\.gate\.weight$")

shard_keep = {}          # shard -> list of final (name, tensor)
new_wm = {}
pruned_tensors = 0
for shard in sorted(set(wm.values())):
    sd = load_file(os.path.join(src, shard))
    out = {}
    for name, t in sd.items():
        m = expert_re.match(name)
        if m:
            # MTP layer index maps to --mtp-drop-from's main-layer drop set
            layer = int(m.group(1))
            if name.startswith(mtp_prefix):
                layer = args.mtp_drop_from
            e = int(m.group(2))
            if e in drop.get(layer, ()):
                pruned_tensors += 1
                continue
            # renumber kept experts contiguously to match compacted router rows
            new_e = remap[layer][e]
            if new_e != e:
                name = expert_re.sub(
                    lambda mm: mm.string[:mm.start(2)] + str(new_e) + ".", name, count=1) \
                    if False else name[:m.start(2)] + str(new_e) + name[m.end(2):]
        g = gate_re.match(name)
        if g:
            layer = int(g.group(2))
            if name.startswith(mtp_prefix):
                layer = args.mtp_drop_from
            t = t[keep[layer], :].contiguous()
        out[name] = t
        new_wm[name] = shard
    shard_keep[shard] = out

os.makedirs(os.path.join(args.dst), exist_ok=True)
for f in os.listdir(src):
    if f.endswith(".safetensors") or f == "model.safetensors.index.json":
        continue
    s = os.path.join(src, f)
    if os.path.isfile(s):
        shutil.copy2(s, os.path.join(args.dst, f))

for shard, tensors in shard_keep.items():
    save_file(tensors, os.path.join(args.dst, shard), metadata={"format": "pt"})

json.dump({"metadata": idx.get("metadata", {}), "weight_map": new_wm},
          open(os.path.join(args.dst, "model.safetensors.index.json"), "w"), indent=2)

cfg = json.load(open(os.path.join(args.dst, "config.json")))
for holder in [cfg, cfg.get("text_config", {})]:
    if isinstance(holder, dict) and holder.get("num_experts") == E:
        holder["num_experts"] = N
json.dump(cfg, open(os.path.join(args.dst, "config.json"), "w"), indent=2)

json.dump({"E": E, "N": N,
           "drop": {str(l): drop[l] for l in range(L)},
           "mtp_drop": sorted(mtp_drop)},
          open(os.path.join(args.dst, "prune_manifest.json"), "w"), indent=2)
print(f"wrote {args.dst}: {pruned_tensors} expert tensors dropped; "
      f"{len(new_wm)} tensors remain; num_experts={N}")
