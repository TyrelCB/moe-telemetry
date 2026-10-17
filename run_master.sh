#!/bin/bash
# Stage 1: prune 160/128 + OOD set. Stage 2: per-model serve cycles
# (replay where missing, OOD probe always). Stage 3: guided scoring.
# Stage 4: guided 192 prune + head-to-head.
set -x
cd ~/projects/moe-telemetry
PY=/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/python
VLLM=/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/vllm
SRC=/home/tyrel/models/FreeToken-Qwen3.5-35B-A3B-FP8

wait_health() { for i in $(seq 1 80); do sleep 15; [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8123/health 2>/dev/null)" = "200" ] && return 0; done; return 1; }
kill_server() { pkill -f "bin/vllm serve"; sleep 20; }
serve() { setsid nohup $VLLM serve $1 --served-model-name $2 --port 8123 --gpu-memory-utilization 0.35 --max-model-len 8192 --enable-return-routed-experts > srv_$2.log 2>&1 < /dev/null & wait_health; }
replay() { $PY collect_routed_experts.py --base-url http://localhost:8123 --model $1 --dataset calib_transcripts.jsonl --out runs/pruned-replay-$1 --max-tokens 256 > runs/replay_$1.log 2>&1; }
probe() { $PY logprob_equivalence.py $2 --url http://localhost:8123 --out $3 --n 32 > probe_$3.log 2>&1; }

# ---- stage 1
[ -d ${SRC}-pruned160 ] || python3 prune_checkpoint.py $SRC runs/tyrel-transcripts/counts.npy ${SRC}-pruned160 --nkeep 160 || exit 1
[ -d ${SRC}-pruned128 ] || python3 prune_checkpoint.py $SRC runs/tyrel-transcripts/counts.npy ${SRC}-pruned128 --nkeep 128 || exit 1
[ -f ood.jsonl ] || python3 make_ood.py || exit 1
python3 make_guided_cands.py runs/tyrel-transcripts/counts.npy 96 > make_cands.log 2>&1
echo STAGE1_DONE

# ---- stage 2: baseline OOD probe
kill_server
serve $SRC base || exit 1
probe base ood.jsonl equiv_ood_base.json
echo STAGE2_BASE_DONE

# ---- stage 2b: OOD probe existing tiers
for N in 241 224 192; do
  serve ${SRC}-pruned$N p$N || exit 1
  probe p$N ood.jsonl equiv_ood_p$N.json
  kill_server
done
echo STAGE2_TIERS_DONE

# ---- stage 2c: new tiers: replay + in-domain + OOD probe
for N in 160 128; do
  serve ${SRC}-pruned$N p$N || exit 1
  replay p$N
  probe p$N calib_transcripts.jsonl equiv_tier$N.json
  probe p$N ood.jsonl equiv_ood_p$N.json
  kill_server
done
echo STAGE2_NEW_DONE

# ---- stage 3: guided chunk scoring on baseline (GPU free now)
$PY ablation_score.py --model $SRC --candidates guided_cands.jsonl \
  --probes calib_transcripts.jsonl --out runs/ablation-guided/scores.json \
  --num-probes 16 --util 0.35 > ablation_guided.log 2>&1
echo STAGE3_DONE

# ---- stage 4: guided drop-set (drop 4 lowest-delta chunks/layer = 32 experts),
# prune, serve, replay, probe both domains
$PY make_guided_drop.py runs/ablation-guided/scores.json guided_cands.jsonl || exit 1
python3 prune_checkpoint.py $SRC runs/tyrel-transcripts/counts.npy ${SRC}-pruned192g \
  --drop-file guided_drop.json || exit 1
serve ${SRC}-pruned192g p192g || exit 1
replay p192g
probe p192g calib_transcripts.jsonl equiv_p192g.json
probe p192g ood.jsonl equiv_ood_p192g.json
kill_server

# ---- final comparisons
for N in 160 128; do
  python3 compare_pruned.py runs/tyrel-transcripts/counts.npy runs/pruned-replay-p$N/counts.npy ${SRC}-pruned$N/prune_manifest.json > tier${N}_compare.json 2>&1
done
python3 compare_pruned.py runs/tyrel-transcripts/counts.npy runs/pruned-replay-p192g/counts.npy ${SRC}-pruned192g/prune_manifest.json > tier192g_compare.json 2>&1
$PY logprob_equivalence.py --diff equiv_base.json equiv_p192g.json > tier192g_equiddiff.json 2>&1
$PY logprob_equivalence.py --diff equiv_ood_base.json equiv_ood_p192g.json > tier192g_ooddiff.json 2>&1
for N in 241 224 192 160 128; do
  $PY logprob_equivalence.py --diff equiv_ood_base.json equiv_ood_p$N.json > ood_p${N}_diff.json 2>&1
done
for N in 160 128; do
  $PY logprob_equivalence.py --diff equiv_base.json equiv_tier$N.json > tier${N}_equiddiff.json 2>&1
done
echo ALL_DONE
