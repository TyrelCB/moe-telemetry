#!/bin/bash
cd ~/projects/moe-telemetry
pkill -f "bin/vllm serve" 2>/dev/null
sleep 8
setsid nohup ./run_phase2.sh > run_phase2.log 2>&1 < /dev/null &
sleep 5
test -f run_phase2.log && echo LAUNCHED || echo FAILED
