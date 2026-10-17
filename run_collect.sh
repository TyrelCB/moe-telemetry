#!/bin/bash
cd ~/projects/moe-telemetry
pkill -f "bin/vllm serve"; sleep 10
/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/python arditi_collect.py --labels refusal_orbench_base.json --out arditi_dirs.pt > collect.log 2>&1
echo done >> collect.log
