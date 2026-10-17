#!/usr/bin/env python3
"""Full Arditi-style two-sided refusal-direction ablation for Qwen3.8-27B-NVFP4.

Per-layer directions d_l extracted from real refusal/compliant activations
(arditi_collect.py). Two-sided edit on the attention surface (all 64 layers,
all on the proven byte-exact FP8 per-output-channel path):
  writer side (o_proj / linear_attn out_proj):  W <- W - d (d^T W)
  reader side (q,k,v / in_proj_qkv,in_proj_z):  W <- W - (W dr)(dr^T),
      dr = gamma_l * d  (direction mapped through input layernorm into
      the post-LN space the projections actually read)
Requant: per-output-channel FP8, amax/448 (same path as the round-trip gate).
The lm_head and MLP NVFP4 tensors are left untouched.
"""
import argparse, json, os, re, shutil
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--dirs", default="arditi_dirs.pt")
ap.add_argument("--factor", type=float, default=1.0, help="1.0=full removal")
ap.add_argument("--sides", default="both", choices=["both","write"])
a = ap.parse_args()

ck = torch.load(a.dirs, map_location="cpu", weights_only=False)
dirs, gamma = ck["dirs"], ck["gamma"]
print("directions for", len(dirs), "layers")

WRITERS = {"self_attn.o_proj", "linear_attn.out_proj"}
READERS = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
           "linear_attn.in_proj_qkv", "linear_attn.in_proj_z"}
TOK_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.((?:self_attn|linear_attn)\.[a-z_0-9]+)\.weight$")

os.makedirs(a.dst, exist_ok=True)
idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
n_w = n_r = 0
for shard in sorted(set(idx["weight_map"].values())):
    sd = load_file(os.path.join(a.src, shard))
    for name in list(sd.keys()):
        m = TOK_RE.match(name)
        if not m: continue
        l, modpath = int(m.group(1)), m.group(2)
        if l not in dirs: continue
        sname = name + "_scale"
        if sname not in sd: continue
        if sd[name].dtype != torch.float8_e4m3fn: continue
        sc = sd[sname].float()
        assert sc.ndim == 2 and sc.shape[1] == 1, name
        w32 = sd[name].float() * sc
        d = dirs[l]

        if modpath in WRITERS:
            w32 = w32 - a.factor * torch.outer(d, d @ w32)
            n_w += 1
        elif modpath in READERS and a.sides == "both":
            dr = gamma[l] * d
            dr = dr / dr.norm()
            w32 = w32 - a.factor * torch.outer(w32 @ dr, dr)
            n_r += 1
        else:
            continue
        finfo = torch.finfo(sd[name].dtype)
        ns = (w32.abs().amax(dim=1, keepdim=True) / finfo.max).clamp_min(1e-12)
        sd[name] = (w32 / ns).to(torch.float8_e4m3fn)
        sd[sname] = ns.to(torch.float8_e4m3fn)
    save_file(sd, os.path.join(a.dst, shard), metadata={"format": "pt"})

for f in os.listdir(a.src):
    if f.endswith(".safetensors") or f == "model.safetensors.index.json": continue
    s = os.path.join(a.src, f)
    if os.path.isfile(s): shutil.copy2(s, os.path.join(a.dst, f))
shutil.copy2(os.path.join(a.src, "model.safetensors.index.json"), a.dst)
json.dump({"method": "arditi two-sided per-layer attention ablation",
           "writers": n_w, "readers": n_r, "factor": a.factor},
          open(os.path.join(a.dst, "obliteration_manifest.json"), "w"), indent=2)
print(f"edited {n_w} writers + {n_r} readers -> {a.dst}")
