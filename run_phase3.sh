#!/bin/bash
# Phase 3: re-run pass@1 with the fixed chat-endpoint script on base + p241.
set -x
cd ~/projects/moe-telemetry
PY=/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/python
VLLM=/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/vllm
SRC=/home/tyrel/models/FreeToken-Qwen3.5-35B-A3B-FP8

wait_health() { for i in $(seq 1 80); do sleep 15; [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8123/health 2>/dev/null)" = "200" ] && return 0; done; return 1; }
kill_server() { pkill -f "bin/vllm serve"; sleep 20; }
serve() { setsid nohup $VLLM serve $1 --served-model-name $2 --port 8123 --gpu-memory-utilization 0.35 --max-model-len 8192 --max-num-seqs 48 > srv2_$2.log 2>&1 < /dev/null & wait_health; }

while ! grep -q PHASE2_ALL_DONE run_phase2.log 2>/dev/null; do sleep 60; done
sleep 10
kill_server
serve $SRC base || exit 1
$PY pass_at_1.py --url http://localhost:8123 --model base --n 24 --out pass1_base.json
kill_server
serve ${SRC}-pruned241 p241 || exit 1
$PY pass_at_1.py --url http://localhost:8123 --model p241 --n 24 --out pass1_p241.json
kill_server
echo PHASE3_DONE
