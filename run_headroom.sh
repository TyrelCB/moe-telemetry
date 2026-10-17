#!/bin/bash
# Serve Qwen3.8-27B-NVFP4 and measure refusal headroom (pre-ablation gate).
set -x
cd ~/projects/moe-telemetry
export PATH=/usr/local/cuda/bin:$PATH
VLLM=/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/vllm
MODEL=/home/tyrel/models/Qwen3.8-27B-NVFP4

pkill -f "bin/vllm serve"; sleep 15
setsid nohup $VLLM serve $MODEL --served-model-name q38 --port 8123 \
  --gpu-memory-utilization 0.40 --max-model-len 8192 --max-num-seqs 48 \
  > srv_q38.log 2>&1 < /dev/null &
for i in $(seq 1 100); do sleep 15
  [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8123/health 2>/dev/null)" = "200" ] && break
done
curl -s -o /dev/null -w "health:%{http_code}\n" localhost:8123/health
/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/python refusal_check.py \
  --url http://localhost:8123 --model q38 --out refusal_q38_base.json
echo HEADROOM_DONE
