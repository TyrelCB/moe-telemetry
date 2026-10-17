#!/usr/bin/env python3
"""Round-trip sanity gate for the NVFP4/FP8 checkpoint.

Copies the checkpoint, dequantizes + requantizes the FULL-ATTENTION-layer
o_proj tensors with NO edit, and writes them back. If serving this copy
matches the base server's logprobs, our requant path is trustworthy and the
real obliteration edit can proceed on the same path.

Usage: roundtrip_gate.py SRC DST [--layers 3,7,...]
"""
import argparse, json, os, shutil
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--full-attn-layers", default="3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63")
args = ap.parse_args()

LAYERS = set(int(x) for x in args.full_attn_layers.split(","))
import re
LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.self_attn\.o_proj\.weight$")

idx = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))
wm = idx["weight_map"]
os.makedirs(args.dst, exist_ok=True)

touched = 0
stats = []
for shard in sorted(set(wm.values())):
    sd = load_file(os.path.join(args.src, shard))
    changed = False
    for name in list(sd.keys()):
        m = LAYER_RE.match(name)
        if not m or int(m.group(1)) not in LAYERS:
            continue
        w = sd[name]
        sname = name + "_scale"
        if w.dtype in (torch.float8_e4m3fn,) and sname in sd:
            # FP8 per-output-channel: dequant w*sc(row), requant from row amax
            w32 = w.float()
            sc = sd[sname].float()          # (H, 1)
            deq = w32 * sc
            finfo = torch.finfo(w.dtype)
            rowamax = deq.abs().amax(dim=1, keepdim=True)
            newscale = (rowamax / finfo.max).clamp_min(1e-12)
            re_q = (deq / newscale).to(w.dtype)
            # error in fp8 payload space at ORIGINAL scale
            err = ((re_q.float() * newscale) - (w32 * sc)).abs().max().item()
            stats.append({"tensor": name, "scale_shape": tuple(sc.shape), "max_deq_err": err})
            # store with ORIGINAL scales if payload unchanged, else new scale
            if err == 0.0:
                pass  # byte-identical requant; keep everything
            sd[name] = re_q
            sd[sname] = newscale.to(sd[sname].dtype)
            changed = True
            touched += 1
        else:
            stats.append({"tensor": name, "dtype": str(w.dtype), "note": "non-fp8 path TBD"})
    save_file(sd, os.path.join(args.dst, shard), metadata={"format": "pt"})

for f in os.listdir(args.src):
    if f.endswith(".safetensors") or f == "model.safetensors.index.json":
        continue
    s = os.path.join(args.src, f)
    if os.path.isfile(s):
        shutil.copy2(s, os.path.join(args.dst, f))
shutil.copy2(os.path.join(args.src, "model.safetensors.index.json"), args.dst)

errs = [s["max_deq_err"] for s in stats if "max_deq_err" in s]
print(json.dumps({"touched": touched,
                  "max_dequant_err": max(errs) if errs else None,
                  "non_fp8_seen": [s for s in stats if "note" in s][:4]}, indent=2))
