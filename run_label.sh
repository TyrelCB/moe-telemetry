#!/bin/bash
cd ~/projects/moe-telemetry
/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/python refusal_check.py --chat --url http://localhost:8123 --model base --prompts orbench_prompts.jsonl --out refusal_orbench_base.json
echo LABEL_DONE >> label.status
