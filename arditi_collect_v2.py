#!/usr/bin/env python3
"""Completion-anchored refusal directions (Arditi/RepE-style).
For each labeled row, teacher-force prompt+completion through the model and
record residual-stream activations at the LAST token. Per layer:
  d = normalize(mean(R) - mean(C))
which cancels prompt-topic variance by construction.
"""
import argparse, json, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/home/tyrel/models/Qwen3.8-27B-NVFP4")
ap.add_argument("--labels", default="refusal_orbench_base.json")
ap.add_argument("--out", default="arditi_dirs_v2.pt")
ap.add_argument("--max-resp-tokens", type=int, default=512)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda:0")
model.eval()

layers = {}
for name, mod in model.named_modules():
    m = re.search(r"layers\.(\d+)$", name)
    if m and (hasattr(mod, "self_attn") or hasattr(mod, "linear_attn")):
        layers[int(m.group(1))] = mod
L = sorted(layers)
print("layers:", len(L))

acts = []
def mk(mod, inputs):
    acts.append(inputs[0][0, -1, :].detach().float().cpu().clone())
for l in L:
    layers[l].register_forward_pre_hook(mk)

rows = json.load(open(a.labels))["rows"]
def enc_row(r):
    msgs = [{"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["text"][:4000]}]
    s = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
    e = tok(s, truncation=True, max_length=2048, return_tensors="pt")
    return e

def run(rs):
    out = []
    for j, r in enumerate(rs):
        acts.clear()
        with torch.no_grad():
            model(**enc_row(r).to(model.device))
        assert len(acts) == len(L)
        out.append(torch.stack(acts))
        if (j+1) % 20 == 0: print(" fwd", j+1)
    return torch.stack(out)

R = run([r for r in rows if r["refused"]])
C = run([r for r in rows if not r["refused"]])
print("R,C:", R.shape, C.shape)

dirs, sep = {}, {}
for li, l in enumerate(L):
    d = R[:, li, :].mean(0) - C[:, li, :].mean(0)
    nd = d.norm()
    if nd < 1e-6: print("WARN degenerate layer", l); continue
    dirs[l] = (d / nd).to(torch.float32)
    # separation sanity: project each activation onto d, compare group means
    pR = (R[:, li, :] @ dirs[l]); pC = (C[:, li, :] @ dirs[l])
    sd = torch.cat([pR, pC]).std()
    sep[l] = float((pR.mean() - pC.mean()).abs() / (sd + 1e-9))

gam = {}
for name, p in model.named_parameters():
    m = re.search(r"layers\.(\d+)\.(input_layernorm|ln_1)\.weight$", name)
    if m: gam[int(m.group(1))] = p.detach().float().cpu()

torch.save({"dirs": dirs, "gamma": gam, "evr": sep,
            "n_refusal": R.shape[0], "n_comply": C.shape[0]}, a.out)
import statistics
print("saved", a.out, "| median separation (SD units):", round(statistics.median(sep.values()), 3))
