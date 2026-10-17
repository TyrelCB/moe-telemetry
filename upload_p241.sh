#!/bin/bash
cd ~/projects/moe-telemetry
/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/hf upload --include "*" \
  theJuice85/FreeToken-Qwen3.5-35B-A3B-FP8-pruned241 \
  /home/tyrel/models/FreeToken-Qwen3.5-35B-A3B-FP8-pruned241 \
  "" > upload_p241.log 2>&1
echo "exit=$?" >> upload_p241.log
