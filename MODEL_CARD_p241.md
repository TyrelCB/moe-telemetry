---
license: apache-2.0
base_model: FreeToken/FreeToken-Qwen3.5-35B-A3B-FP8
tags:
  - fp8
  - moe
  - pruning
  - quantized
---

# FreeToken-Qwen3.5-35B-A3B-FP8-pruned241

Expert-pruned variant of FreeToken-Qwen3.5-35B-A3B-FP8: **46% of routed experts
removed** (241 of 512 kept per layer, selected by routing-count calibration),
with **no requantization** — surviving FP8 tensors and block scales are
byte-identical to the base. Router rows compacted and renumbered; MTP block
pruned consistently.

## Evidence: behaviorally identical to base

Teacher-forced logprob-equivalence (32 samples/domain, mean per-token delta vs
base server; negative = worse):

| Probe domain | mean delta | worst sample |
|---|---|---|
| In-domain (calibration mix) | -0.01 | — |
| SWE (real repo files) | **-0.009** | -0.146 |
| Tools (OpenAI-style schemas) | **+0.006** | -0.075 |

HumanEval pass@1 (temp 0, thinking off): **14/24, identical to base, same 14 tasks**.

For context, deeper tiers degrade measurably: at 63% pruning (192 experts) OOD
deltas grow ~4x while in-domain still looks clean (calibration-coverage
failure); at 75% pruning every sample worsens. 241 is the safe cut.

## Serving notes

Hybrid-attention model: under vLLM use `--max-num-seqs 48` (or lower) — Mamba
state cache needs one block per sequence and default 256 fails engine init.
FP8 block-128 weights, dynamic activation scale, same loader/kernels as base.

## Method / reproduction

Calibration = routing counts from replayed transcripts; prune = drop
lowest-usage experts per layer, compact router, renumber expert keys. Eval
harness (probes, diff tool, pass-at-1 script) in the study repo:
https://github.com/TyrelCB/moe-telemetry (calibration replay, tier builder, equivalence probes, diff tool, pass-at-1 harness).
