#!/bin/bash
cd /home/tyrel/projects/moe-telemetry
/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/python arditi_collect_v2.py --out arditi_dirs_v2.pt > collect2.log 2>&1
echo done >> collect2.log
