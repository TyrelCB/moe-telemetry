#!/usr/bin/env python3
"""Refusal-direction ablation (obliteration-lite) on the FP8 checkpoint.

Hypothesis (Arditi et al., 2024): refusal is a single direction in the
residual stream written by attention output projections; deleting it
de-censors while preserving most capability. Their method is SVD on a
refusal-prompt covariance; this lite version uses the synthetic ' no'-vs-
' yes' logit-difference direction projected into each o_proj's output space
(the direction whose amplification the model reads as refusal) and zeroes its
component out of every FULL-ATTENTION-layer o_proj.

FP8 gotcha: weights are float8_e4m3fn with block scales (weight_scale_inv).
We dequantize, edit in bf16, requantize per 128x128 block.

Usage: obliterate.py SRC DST [--layers 3,7,...] [--mode zero|soft05]
"""
import argparse, json, os, re, shutil
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--full-attn-layers", default="3,7,11,15,19,23,27,31,35,39")
ap.add_argument("--mode", default="zero", help="zero = remove component; soft05 = damp to 5%")
args = ap.parse_args()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.src)

LAYERS = set(int(x) for x in args.full_attn_layers.split(","))
FACTOR = 0.0 if args.mode == "zero" else 0.05

idx = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))
wm = idx["weight_map"]
# direction: lm_head rows for 'no' minus 'yes' (residual-stream proxy);
# fall back to embedding rows if lm_head is quantized/absent
def row_of(key):
    t = load_file(os.path.join(args.src, wm[key]))[key]
    return t.float()
try:
    lk = next(k for k in wm if k.endswith("lm_head.weight"))
    W = row_of(lk)
    t_yes = W[tok.encode(" yes", add_special_tokens=False)[-1]]
    t_no = W[tok.encode(" no", add_special_tokens=False)[-1]]
    src_dir = "lm_head"
except StopIteration:
    ek = next(k for k in wm if k.endswith("embed_tokens.weight"))
    W = row_of(ek)
    t_yes = W[tok.encode(" yes", add_special_tokens=False)[-1]]
    t_no = W[tok.encode(" no", add_special_tokens=False)[-1]]
    src_dir = "embed_tokens"
d = (t_no - t_yes).to(torch.bfloat16).float()
d = d / d.norm()
del W

expert_re = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
os.makedirs(args.dst, exist_ok=True)
edited = 0
for shard in sorted(set(wm.values())):
    sd = load_file(os.path.join(args.src, shard))
    changed = False
    for name in list(sd.keys()):
        m = re.match(r"^model\.language_model\.layers\.(\d+)\.self_attn\.o_proj\.weight$", name)
        if not m or int(m.group(1)) not in LAYERS:
            continue
        w = sd[name]                       # fp8
        sname = name + "_scale_inv"
        w16 = w.dequantize() if hasattr(w, "dequantize") else w.float()
        w32 = w16.float()
        # remove the refusal component from each COLUMN (residual stream is the
        # OUTPUT side of o_proj: weight shape (out=hidden, in=heads)):
        # w <- w - k * d (d^T w) ; k=1 removes the direction entirely (zero mode),
        # k=0.95 damps its component to 5% (soft05 mode)
        k = 1.0 - FACTOR
        comp = d.to(w32.device) @ w32       # (in,)
        w32 = w32 - k * torch.outer(d.to(w32.device), comp)
        # requantize fp8 block-128
        try:
            wq = w32.to(w.dtype)           # e4m3 cast
            # keep original block scales (direction edit is small-magnitude;
            # recompute scale per 128x128 block to be safe)
            sc = sd[sname]
            H, W = w32.shape
            blocks = w32.reshape(H // 128, 128, W // 128, 128).abs()
            amax = blocks.permute(0, 2, 1, 3).amax(dim=(-1, -2))
            finfo = torch.finfo(w.dtype)
            newscale = (amax / finfo.max).clamp_min(1e-12)
            if tuple(newscale.shape) == tuple(sc.shape):
                sd[sname] = newscale.to(sc.dtype)
            sd[name] = (w32 / newscale.repeat_interleave(128, 0).repeat_interleave(128, 1)
                        ).to(w.dtype)
        except Exception as e:
            print("quant fail", name, e)
            continue
        edited += 1
        changed = True
    if changed or True:
        save_file(sd, os.path.join(args.dst, shard), metadata={"format": "pt"})

for f in os.listdir(args.src):
    if f.endswith(".safetensors") or f == "model.safetensors.index.json":
        continue
    s = os.path.join(args.src, f)
    if os.path.isfile(s):
        shutil.copy2(s, os.path.join(args.dst, f))

json.dump({"method": "refusal-direction o_proj ablation (lite obliteration)",
           "layers": sorted(LAYERS), "mode": args.mode,
           "direction": f"embed[{t_no}]-embed[{t_yes}] normalized",
           "edited_tensors": edited},
          open(os.path.join(args.dst, "obliteration_manifest.json"), "w"), indent=2)
print(f"edited {edited} o_proj tensors -> {args.dst}")
