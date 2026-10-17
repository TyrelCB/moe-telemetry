#!/usr/bin/env python3
"""Refusal-direction ablation for Qwen3.8-27B-NVFP4 (per-output-channel FP8).

Same math as obliterate.py (W <- W - k * d (d^T W) on full-attention o_proj),
but:
  - scales are weight_scale (H,1) per-output-channel, NOT block scales
  - lm_head is quantized here: multiply its rows by their scale before
    forming the direction
  - full-attention layers are every 4th layer: 3,7,...,63

Usage: obliterate_q38.py SRC DST [--mode zero|soft05]
"""
import argparse, json, os, re, shutil
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--full-attn-layers", default=",".join(str(l) for l in range(3, 64, 4)))
ap.add_argument("--mode", default="zero")
args = ap.parse_args()

LAYERS = set(int(x) for x in args.full_attn_layers.split(","))
FACTOR = 0.0 if args.mode == "zero" else 0.05
tok = AutoTokenizer.from_pretrained(args.src)

idx = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))
wm = idx["weight_map"]

def load_key(key):
    return load_file(os.path.join(args.src, wm[key]))[key]

# direction from lm_head rows for ' no' vs ' yes', dequantized properly
lk = next(k for k in wm if k.endswith("lm_head.weight"))
W = load_key(lk).float()
sk = lk + "_scale"
if sk in wm:
    sc = load_key(sk).float()
    if sc.ndim == 2 and sc.shape[1] == 1:
        W = W * sc                      # per-row scale
    else:
        W = W * sc                      # broadcast (per-tensor or block)
t_yes = W[tok.encode(" yes", add_special_tokens=False)[-1]]
t_no = W[tok.encode(" no", add_special_tokens=False)[-1]]
d = (t_no - t_yes)
d = (d / d.norm()).to(torch.float32)
del W
print("direction from lm_head rows (no vs yes token ids), normalized ok")

LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.self_attn\.o_proj\.weight$")
os.makedirs(args.dst, exist_ok=True)
edited = 0
for shard in sorted(set(wm.values())):
    sd = load_file(os.path.join(args.src, shard))
    for name in list(sd.keys()):
        m = LAYER_RE.match(name)
        if not m or int(m.group(1)) not in LAYERS:
            continue
        w = sd[name]
        sname = name + "_scale"
        assert w.dtype == torch.float8_e4m3fn and sname in sd, f"unexpected {name}"
        sc = sd[sname].float()          # (H,1)
        w32 = w.float() * sc            # dequantized
        k = 1.0 - FACTOR
        comp = d @ w32                  # (in,)
        w32 = w32 - k * torch.outer(d, comp)
        finfo = torch.finfo(w.dtype)
        rowamax = w32.abs().amax(dim=1, keepdim=True)
        newscale = (rowamax / finfo.max).clamp_min(1e-12)
        sd[name] = (w32 / newscale).to(w.dtype)
        sd[sname] = newscale.to(sd[sname].dtype)
        edited += 1
    save_file(sd, os.path.join(args.dst, shard), metadata={"format": "pt"})

for f in os.listdir(args.src):
    if f.endswith(".safetensors") or f == "model.safetensors.index.json":
        continue
    s = os.path.join(args.src, f)
    if os.path.isfile(s):
        shutil.copy2(s, os.path.join(args.dst, f))
shutil.copy2(os.path.join(args.src, "model.safetensors.index.json"), args.dst)

json.dump({"method": "refusal-direction o_proj ablation (per-channel FP8)",
           "layers": sorted(LAYERS), "mode": args.mode,
           "edited_tensors": edited},
          open(os.path.join(args.dst, "obliteration_manifest.json"), "w"), indent=2)
print(f"edited {edited} o_proj tensors -> {args.dst}")
