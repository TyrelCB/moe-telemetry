#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
pkill -f "bin/vllm serve"; sleep 8
setsid nohup /home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/vllm serve /home/tyrel/models/Qwen3.8-27B-NVFP4 \
  --served-model-name base --port 8123 --gpu-memory-utilization 0.40 --max-model-len 8192 \
  --max-num-seqs 48 > srv_base_label.log 2>&1 < /dev/null &
