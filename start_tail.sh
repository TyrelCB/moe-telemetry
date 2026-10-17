#!/bin/bash
cd ~/projects/moe-telemetry || exit 1
# kill any existing drivers by pattern (safe here: our own cmdline is './start_tail.sh')
for pat in run_phase2 run_phase3; do
  pkill -f "$pat" 2>/dev/null
done
sleep 3
chmod +x run_phase2c.sh run_phase3.sh
setsid nohup ./run_phase2c.sh > run_phase2c.log 2>&1 < /dev/null &
sleep 2
setsid nohup ./run_phase3.sh > run_phase3.log 2>&1 < /dev/null &
sleep 5
pgrep -af 'bash ./run_phase' | head -4
