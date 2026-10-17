#!/bin/bash
cd ~/projects/moe-telemetry
PY=/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/python
VLLM=/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/vllm
wait_health() { for i in $(seq 1 80); do sleep 15; [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8123/health 2>/dev/null)" = "200" ] && return 0; done; return 1; }
kill_server() { pkill -f "vllm serve"; sleep 15; }

kill_server
setsid nohup $VLLM serve /home/tyrel/models/FreeToken-Qwen3.5-35B-A3B-FP8 --served-model-name base --port 8123 --gpu-memory-utilization 0.66 --max-model-len 8192 > equiv_base.log 2>&1 < /dev/null &
wait_health || { echo "base server failed"; exit 1; }
$PY logprob_equivalence.py calib_transcripts.jsonl --url http://localhost:8123 --out equiv_base.json --n 32

kill_server
setsid nohup $VLLM serve /home/tyrel/models/FreeToken-Qwen3.5-35B-A3B-FP8-pruned241 --served-model-name pruned241 --port 8123 --gpu-memory-utilization 0.66 --max-model-len 8192 > equiv_pruned.log 2>&1 < /dev/null &
wait_health || { echo "pruned server failed"; exit 1; }
$PY logprob_equivalence.py calib_transcripts.jsonl --url http://localhost:8123 --out equiv_pruned.json --n 32

$PY logprob_equivalence.py --diff equiv_base.json equiv_pruned.json > equiv_diff.json 2>&1
echo EQUIV_DONE
