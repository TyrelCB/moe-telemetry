#!/usr/bin/env python3
"""Dense-model telemetry collector (HF transformers, GPU).

Dense FFNs have no expert routing; the analogues captured here:
  1. FFN neuron activation stats (mean |act|, dead fraction, gini) —
     forward hooks on each MLP down_proj input. Width-pruning signal.
  2. Per-head attention entropy on full-attention layers (eager mode)
     — the "who does this token consult" routing analogue.

Works on hybrid stacks (Qwen3.5 dense: 3 gated-delta + 1 full attn per 4).
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--attn", action="store_true", help="capture attention entropy (eager)")
    args = ap.parse_args()

    texts = []
    with open(args.dataset) as f:
        for line in f:
            if len(texts) >= args.num_samples:
                break
            t = json.loads(line)["messages"][0]["content"].strip()
            if len(t) >= 80:
                texts.append(t)
    print(f"{len(texts)} samples", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager" if args.attn else "sdpa",
        trust_remote_code=True)
    model.eval()

    # locate decoder layers regardless of wrapper depth
    layers = None
    for attr_path in (("model", "layers"), ("language_model", "model", "layers"),
                      ("model", "model", "layers")):
        obj = model
        ok = True
        for a in attr_path:
            obj = getattr(obj, a, None)
            if obj is None:
                ok = False
                break
        if ok:
            layers = obj
            break
    if layers is None:
        sys.exit("could not locate decoder layers")
    L = len(layers)
    cfg = getattr(model.config, "text_config", model.config)
    Inter = cfg.intermediate_size
    print(f"L={L} inter={Inter}", flush=True)

    counts = np.zeros((L, Inter))
    zero_counts = np.zeros((L, Inter), dtype=np.int64)
    tok_count = {"n": 0}
    def mk_hook(li):
        def hook(module, inputs, output=None):
            x = inputs[0].reshape(-1, inputs[0].shape[-1]).float()
            counts[li] += x.abs().mean(dim=0).cpu().numpy()
            zero_counts[li] += (x.abs() < 1e-6).sum(dim=0).cpu().numpy()
        return hook
    handles = [layers[li].mlp.down_proj.register_forward_hook(mk_hook(li)) for li in range(L)]

    attn_entropy = [[] for _ in range(L)]
    layer_types = getattr(cfg, "layer_types", None)
    full_attn_idx = [i for i, t in enumerate(layer_types)] if layer_types else list(range(L))
    if layer_types:
        full_attn_idx = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    t0 = time.time()
    with torch.no_grad():
        for i, t in enumerate(texts):
            enc = tok(t, truncation=True, max_length=args.max_len, return_tensors="pt").to("cuda")
            out = model(**enc, output_attentions=args.attn)
            tok_count["n"] += enc["input_ids"].shape[1]
            if args.attn and getattr(out, "attentions", None):
                # hybrid: attentions tuple covers full-attention layers only
                aw_list = [a for a in out.attentions if a is not None and a.numel() > 0 and a.shape[1] > 0]
                for slot, aw in enumerate(aw_list):
                    li = full_attn_idx[slot] if slot < len(full_attn_idx) else slot
                    p = torch.softmax(aw.float(), dim=-1)
                    ent = -(p * (p + 1e-9).log()).sum(-1).mean(dim=(0, 2)).cpu().numpy()
                    attn_entropy[li].append(ent)
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(texts)}] {time.time()-t0:.0f}s", flush=True)
    for h in handles:
        h.remove()

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "neuron_counts.npy"), counts)
    np.save(os.path.join(args.out, "neuron_zero_frac.npy"), zero_counts / max(tok_count["n"], 1))
    frac_dead = zero_counts / max(tok_count["n"], 1)
    gs = []
    for li in range(L):
        x = np.sort(counts[li]); n = len(x)
        gs.append(float(((2 * np.arange(1, n + 1) - n - 1) * x).sum() / (n * x.sum())) if x.sum() else 0.0)
    stats = {"model": args.model, "samples": len(texts), "tokens": tok_count["n"],
             "layers": L, "intermediate": Inter,
             "dead_neurons_ge99pct_off": int((frac_dead >= 0.99).sum()),
             "dead_by_layer": {str(li): int((frac_dead[li] >= 0.99).sum()) for li in range(L)},
             "gini_mean_act": gs}
    with open(os.path.join(args.out, "neuron_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    if args.attn:
        aw_stats = {}
        for li in range(L):
            if attn_entropy[li]:
                arr = np.stack(attn_entropy[li])
                aw_stats[str(li)] = {"mean_entropy_per_head": arr.mean(0).round(4).tolist()}
        with open(os.path.join(args.out, "attn_stats.json"), "w") as f:
            json.dump(aw_stats, f, indent=2)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"argv": sys.argv[1:], "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f, indent=2)
    print(f"wrote {args.out}: dead(>=99% off)={stats['dead_neurons_ge99pct_off']} gini={np.mean(gs):.3f}")

if __name__ == "__main__":
    main()
