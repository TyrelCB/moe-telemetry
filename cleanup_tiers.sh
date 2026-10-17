#!/bin/bash
pkill -f "run_tiers" 2>/dev/null
pkill -f "collect_routed_experts" 2>/dev/null
pkill -f "bin/vllm serve" 2>/dev/null
sleep 15
rm -rf ~/projects/moe-telemetry/runs/pruned-replay-224
echo cleanup-ok
