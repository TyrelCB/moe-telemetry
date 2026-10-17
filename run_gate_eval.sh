#!/bin/bash
# Serve the round-trip gate copy; logprob-diff vs base. Gate = near-zero delta.
set -x
cd ~/projects/moe-telemetry
export PATH=/usr/local/cuda/bin:$PATH
PY=/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/python
VLLM=/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/vllm

pkill -f "bin/vllm serve"; sleep 15
setsid nohup $VLLM serve /home/tyrel/models/Qwen3.8-27B-NVFP4-gate --served-model-name q38g \
  --port 8123 --gpu-memory-utilization 0.40 --max-model-len 8192 --max-num-seqs 48 \
  > srv_gate.log 2>&1 < /dev/null &
for i in $(seq 1 100); do sleep 15
  [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8123/health 2>/dev/null)" = "200" ] && break
done
$PY logprob_equivalence.py calib_transcripts.jsonl --url http://localhost:8123 --out equiv_q38_gate.json --n 32 > probe_gate.log 2>&1
$PY logprob_equivalence.py --diff equiv_q38_base.json equiv_q38_gate.json > gate_diff.json 2>&1
cat gate_diff.json
echo GATE_EVAL_DONE
