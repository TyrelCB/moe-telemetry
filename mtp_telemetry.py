#!/usr/bin/env python3
"""Speculative-decoding (MTP head) routable telemetry probe.

Captures /metrics before/after a request burst:
  - vllm:spec_decode_num_drafts_total
  - vllm:spec_decode_num_draft_tokens_total
  - vllm:spec_decode_num_accepted_tokens_total
  - vllm:spec_decode_num_accepted_tokens_per_pos{position=...}
and checks whether routed-expert traces grow a layer when the MTP head's
MoE runs (mtp.layers.0.mlp is itself an MoE).
"""
import base64, io, json, re, sys, urllib.request
import numpy as np

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8124"
MODEL = sys.argv[2]

def metrics():
    txt = urllib.request.urlopen(BASE + "/metrics").read().decode()
    out = {}
    def tot(pat):
        return sum(int(float(v)) for v in re.findall(pat, txt))
    out["drafts"] = tot(r"vllm:spec_decode_num_drafts_total\{[^}]*\}\s+([\d.e+]+)")
    out["draft_tokens"] = tot(r"vllm:spec_decode_num_draft_tokens_total\{[^}]*\}\s+([\d.e+]+)")
    out["accepted"] = tot(r"vllm:spec_decode_num_accepted_tokens_total\{[^}]*\}\s+([\d.e+]+)")
    pp = {}
    for pos, v in re.findall(
            r'vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"[^}]*\}\s+([\d.e+]+)', txt):
        pp[int(pos)] = pp.get(int(pos), 0) + int(float(v))
    out["per_pos"] = [pp[k] for k in sorted(pp)]
    return out

def post(msgs):
    payload = {"model": MODEL, "messages": msgs, "max_tokens": 64, "temperature": 0,
               "return_routed_experts": True}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))

prompts = [t for t in (json.loads(l)["messages"][0]["content"] for l in open(sys.argv[3])) if len(t) >= 80][:24]
before = metrics()
shapes = []
for p in prompts:
    r = post([{"role": "user", "content": p}])
    ch = r["choices"][0]
    if ch.get("routed_experts"):
        arr = np.load(io.BytesIO(base64.b64decode(ch["routed_experts"])))
        shapes.append(arr.shape)
after = metrics()
print(json.dumps({
    "requests": len(prompts),
    "metrics_before": before, "metrics_after": after,
    "delta": {k: after[k] - before[k] for k in ("drafts", "draft_tokens", "accepted")},
    "per_pos_delta": [b - a for a, b in zip(before["per_pos"], after["per_pos"])] if len(before["per_pos"]) == len(after["per_pos"]) else after["per_pos"],
    "trace_shapes": sorted(set(map(str, shapes))),
}, indent=2))
d_drafts = after["drafts"] - before["drafts"]
d_acc = after["accepted"] - before["accepted"]
d_dtok = after["draft_tokens"] - before["draft_tokens"]
if d_drafts and d_dtok:
    print(f"\nspec-head acceptance: drafts={d_drafts} draft_tokens={d_dtok} accepted={d_acc} "
          f"accept_rate={d_acc/d_dtok:.3f} mean_accepted/draft={d_acc/d_drafts:.2f}")
