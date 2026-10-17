#!/usr/bin/env python3
"""Expert-ablation importance scorer for vLLM MoE (offline, in-process).

Candidates: JSONL {layer, expert[, chunk]} or summary.json (dead_experts).
Scoring unit = (layer, chunk): all experts sharing a chunk within a layer are
masked together in ONE pass (chunk absent = 0 = one group per layer).
Measures teacher-forced logprob/token degradation over a probe set.

Gotchas (learned the hard way): engine must be in-process
(VLLM_ENABLE_V1_MULTIPROCESSING=0 + uni backend); mask delivered via JSON file
with mtime poll (fork doesn't share mutable state); mask file + out dir must
exist BEFORE LLM() because warmup invokes the patch.
"""
import argparse, json, os, sys, time
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidates", required=True, help="summary.json or JSONL {layer,expert[,chunk]}")
    ap.add_argument("--probes", required=True, help="calibration JSONL used by collector")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=32, help="scored tail per probe")
    ap.add_argument("--num-probes", type=int, default=16)
    ap.add_argument("--util", type=float, default=0.66)
    args = ap.parse_args()

    recs = []
    with open(args.candidates) as f:
        txt = f.read()
    try:
        d = json.loads(txt)
        for l, e in d.get("dead_experts", []):
            recs.append((l, e, 0))
    except json.JSONDecodeError:
        for line in txt.splitlines():
            r = json.loads(line)
            recs.append((int(r["layer"]), int(r["expert"]), int(r.get("chunk", 0))))
    if not recs:
        sys.exit("no candidates")
    units = sorted({(l, c) for l, _, c in recs})
    unit_experts = {(l, c): sorted(e for ll, e, cc in recs if ll == l and cc == c)
                    for l, c in units}
    print(f"{len(recs)} candidates, {len(units)} scoring units", flush=True)

    probes = []
    with open(args.probes) as f:
        for line in f:
            if len(probes) >= args.num_probes:
                break
            t = json.loads(line)["messages"][0]["content"].strip()
            if len(t) < 120:
                continue
            mid = len(t) // 2
            probes.append((t[:mid], t[mid:]))
    print(f"{len(probes)} probes", flush=True)

    from vllm import LLM, SamplingParams
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    mask_path = os.path.join(os.path.dirname(os.path.abspath(args.out)) or ".", "mask.json")
    with open(mask_path, "w") as f:
        f.write("{}")
    llm = LLM(model=args.model, max_model_len=4096, gpu_memory_utilization=args.util,
              enable_prefix_caching=False, enable_return_routed_experts=False,
              disable_log_stats=True, enforce_eager=True,
              distributed_executor_backend="uni")
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)

    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    orig_router = BaseRouter.select_experts
    orig_apply = MoERunner._apply_quant_method

    def _apply(self, hidden_states, router_logits, shared_experts_input, input_ids=None):
        import builtins
        builtins._ablate_cur_layer = self.layer_id
        return orig_apply(self, hidden_states, router_logits, shared_experts_input, input_ids=input_ids)

    _cache = {"mtime": 0, "mask": {}}

    def _patched(self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
        w, ids = orig_router(self, hidden_states, router_logits,
                             topk_indices_dtype=topk_indices_dtype, input_ids=input_ids)
        import builtins
        lid = getattr(builtins, "_ablate_cur_layer", None)
        try:
            st = os.stat(mask_path)
            if st.st_mtime != _cache["mtime"]:
                _cache["mtime"] = st.st_mtime
                with open(mask_path) as fh:
                    _cache["mask"] = {int(k): v for k, v in json.load(fh).items()}
                with open(mask_path + ".hits", "a") as fh:
                    fh.write("reset\n")
        except OSError:
            return w, ids
        m = _cache["mask"].get(lid) if lid is not None else None
        if m:
            mt = torch.as_tensor(m, dtype=ids.dtype, device=ids.device)
            hit = torch.isin(ids.reshape(-1), mt).view(ids.shape)
            w = w.masked_fill(hit, 0.0)
            with open(mask_path + ".hits", "a") as fh:
                fh.write(f"{lid} {int(hit.sum())}\n")
        return w, ids

    BaseRouter.select_experts = _patched
    MoERunner._apply_quant_method = _apply

    def score_mask(spec):
        with open(mask_path + ".hits", "w") as fh:
            fh.write("")
        with open(mask_path, "w") as fh:  # mtime bump propagates to worker
            json.dump({str(l): list(es) for l, es in spec.items()}, fh)
        outs = llm.generate([p + c for p, c in probes], sp)
        hits = 0
        try:
            with open(mask_path + ".hits") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) == 2:
                        hits += int(parts[1])
        except OSError:
            pass
        totals = []
        for o in outs:
            ids = o.prompt_token_ids
            plps = o.prompt_logprobs
            k = min(args.max_tokens, len(ids))
            s = 0.0
            for i in range(len(ids) - k, len(ids)):
                row = plps[i]
                if row and ids[i] in row:
                    s += row[ids[i]].logprob
            totals.append(s / max(k, 1))
        return float(np.mean(totals)), hits

    t0 = time.time()
    base, _ = score_mask({})
    print(f"baseline mean logprob/token: {base:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    results = []
    for u in units:
        exps = unit_experts[u]
        d, hits = score_mask({u[0]: exps})
        results.append({"layer": u[0], "chunk": u[1], "experts": exps,
                        "mean_dlogprob_token": base - d, "masked_slot_hits": hits})
        print(f"layer {u[0]} chunk {u[1]} ({len(exps)} experts): d_logprob={base-d:+.5f} "
              f"hits={hits} ({time.time()-t0:.0f}s)", flush=True)

    results.sort(key=lambda r: -r["mean_dlogprob_token"])
    with open(args.out, "w") as f:
        json.dump({"baseline_logprob_token": base, "results": results}, f, indent=2)
    BaseRouter.select_experts = orig_router
    MoERunner._apply_quant_method = orig_apply
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
