#!/bin/bash
cd ~/projects/moe-telemetry || exit 1
pkill -f "run_phase2" 2>/dev/null
pkill -f "run_phase3" 2>/dev/null
sleep 3
chmod +x run_phase4.sh
setsid nohup ./run_phase4.sh > run_phase4.log 2>&1 < /dev/null &
sleep 5
pgrep -af 'bash ./run_phase4' | head -2
