#!/usr/bin/env python3
"""Collect per-layer residual-stream activations on refusal vs compliant prompts,
extract per-layer refusal directions (Arditi et al.). Loads the compressed-tensors
checkpoint directly in transformers (verified: Qwen3_5ForCausalLM loads natively).
"""
import argparse, json, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/home/tyrel/models/Qwen3.8-27B-NVFP4")
ap.add_argument("--labels", required=True)
ap.add_argument("--out", default="arditi_dirs.pt")
ap.add_argument("--max-prompts", type=int, default=60)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda:0")
model.eval()

layers = {}
for name, mod in model.named_modules():
    m = re.search(r"layers\.(\d+)$", name)
    if m and (hasattr(mod, "self_attn") or hasattr(mod, "linear_attn")):
        layers[int(m.group(1))] = (name, mod)
L = sorted(layers)
print("decoder layers found:", len(L), L[0], "..", L[-1])

acts = []
def mk(i):
    def fn(mod, inputs):
        acts.append(inputs[0][0, -1, :].detach().float().cpu().clone())
    return fn
for i in L:
    layers[i][1].register_forward_pre_hook(mk(i))

data = json.load(open(a.labels))
items = data if isinstance(data, list) else data.get("results") or data.get("items") or data.get("rows", [])
refusal = [x["prompt"] for x in items if x.get("refused")]
comply  = [x["prompt"] for x in items if not x.get("refused")]
refusal, comply = refusal[:a.max_prompts], comply[:a.max_prompts]
print("refusal set:", len(refusal), "compliant set:", len(comply))

def run(prompts):
    out = []
    for j, p in enumerate(prompts):
        s = tok.apply_chat_template([{"role": "user", "content": p}],
                                    add_generation_prompt=True, tokenize=False)
        enc = tok(s, return_tensors="pt").to(model.device)
        acts.clear()
        with torch.no_grad():
            model(**enc)
        assert len(acts) == len(L), f"hook count {len(acts)} != {len(L)}"
        out.append(torch.stack(acts))          # (n_layers, H)
        if (j + 1) % 10 == 0: print("  fwd", j + 1)
    return torch.stack(out)                    # (P, n_layers, H)

R = run(refusal); C = run(comply)

# per-layer direction: top PC of mean-centered concat, anchored on mean-diff sign
dirs, evr = {}, {}
for li, l in enumerate(L):
    Rl, Cl = R[:, li, :], C[:, li, :]
    mu = torch.cat([Rl, Cl]).mean(0)
    X = torch.cat([Rl, Cl]) - mu
    U, S, V = torch.linalg.svd(X, full_matrices=False)
    d = V[0]
    if (d @ (Rl.mean(0) - Cl.mean(0))) < 0: d = -d
    d = d / d.norm()
    dirs[l] = d.to(torch.float32)
    evr[l] = float((S[0] ** 2 / (S ** 2).sum()))

# layernorm gammas for reader-space mapping
gam = {}
for name, p in model.named_parameters():
    m = re.search(r"layers\.(\d+)\.input_layernorm\.weight$|layers\.(\d+)\.ln_1\.weight$", name)
    if m:
        l = int(m.group(1) or m.group(2))
        gam[l] = p.detach().float().cpu()

torch.save({"dirs": dirs, "gamma": gam, "evr": evr,
            "n_refusal": len(refusal), "n_comply": len(comply)}, a.out)
import statistics
print("saved", a.out, "| median explained-var-ratio:", round(statistics.median(evr.values()), 3))
