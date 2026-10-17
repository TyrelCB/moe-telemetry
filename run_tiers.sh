#!/bin/bash
# Prune-harder tiers: keep 224 and 192 experts/layer, then replay + probe.
set -x
cd ~/projects/moe-telemetry
PY=/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/python
VLLM=/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/vllm
SRC=/home/tyrel/models/FreeToken-Qwen3.5-35B-A3B-FP8

wait_health() { for i in $(seq 1 80); do sleep 15; [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8123/health 2>/dev/null)" = "200" ] && return 0; done; return 1; }
kill_server() { pkill -f "bin/vllm serve"; sleep 20; }

kill_server
for N in 224 192; do
  DST=${SRC}-pruned$N
  [ -d $DST ] || python3 prune_checkpoint.py $SRC runs/tyrel-transcripts/counts.npy $DST --nkeep $N || { echo "PRUNE_FAIL $N"; exit 1; }
done

for N in 224 192; do
  DST=${SRC}-pruned$N
  setsid nohup $VLLM serve $DST --served-model-name p$N --port 8123 --gpu-memory-utilization 0.66 --max-model-len 8192 --enable-return-routed-experts > tier_$N.log 2>&1 < /dev/null &
  wait_health || { echo "SERVE_FAIL $N"; exit 1; }
  $PY collect_routed_experts.py --base-url http://localhost:8123 --model p$N --dataset calib_transcripts.jsonl --out runs/pruned-replay-$N --max-tokens 256 > runs/pruned-replay-$N.collect.log 2>&1
  $PY logprob_equivalence.py calib_transcripts.jsonl --url http://localhost:8123 --out equiv_tier$N.json --n 32 > tier${N}_probe.log 2>&1
  kill_server
done

for N in 224 192; do
  python3 compare_pruned.py runs/tyrel-transcripts/counts.npy runs/pruned-replay-$N/counts.npy ${SRC}-pruned$N/prune_manifest.json > tier${N}_compare.json 2>&1
  $PY logprob_equivalence.py --diff equiv_base.json equiv_tier$N.json > tier${N}_equiddiff.json 2>&1
done
echo TIER_DONE
