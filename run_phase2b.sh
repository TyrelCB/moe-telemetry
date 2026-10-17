#!/bin/bash
# Phase 2B: resume after obliterate.py fix — build oblit ckpt, eval it, diffs.
set -x
cd ~/projects/moe-telemetry
PY=/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/python
VLLM=/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/vllm
SRC=/home/tyrel/models/FreeToken-Qwen3.5-35B-A3B-FP8

wait_health() { for i in $(seq 1 80); do sleep 15; [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8123/health 2>/dev/null)" = "200" ] && return 0; done; return 1; }
kill_server() { pkill -f "bin/vllm serve"; sleep 20; }
serve() { setsid nohup $VLLM serve $1 --served-model-name $2 --port 8123 --gpu-memory-utilization 0.35 --max-model-len 8192 --max-num-seqs 128 > srv2_$2.log 2>&1 < /dev/null & wait_health; }
probe() { $PY logprob_equivalence.py $2 --url http://localhost:8123 --out $3 --n 32 > probe_$3.log 2>&1; }
passat() { $PY pass_at_1.py --url http://localhost:8123 --model $1 --n 24 --out pass1_$1.json; }

if [ ! -f ${SRC}-oblit/obliteration_manifest.json ]; then
  rm -rf ${SRC}-oblit
  $PY obliterate.py $SRC ${SRC}-oblit || { echo OBLIT_BUILD_FAILED; exit 1; }
fi
cp $SRC/model.safetensors.index.json ${SRC}-oblit/

kill_server
serve ${SRC}-oblit oblit || exit 1
$PY refusal_check.py --url http://localhost:8123 --out refusal_oblit.json
probe oblit calib_transcripts.jsonl equiv_oblit_id.json
probe oblit ood2_swe.jsonl equiv_swe_oblit.json
probe oblit ood2_tools.jsonl equiv_tools_oblit.json
passat oblit
kill_server

serve $SRC base || exit 1
$PY refusal_check.py --url http://localhost:8123 --out refusal_base.json
kill_server

for T in p241 p224 p192 p128 oblit; do
  $PY logprob_equivalence.py --diff equiv_swe_base.json equiv_swe_$T.json > swe_${T}_diff.json 2>&1
  $PY logprob_equivalence.py --diff equiv_tools_base.json equiv_tools_$T.json > tools_${T}_diff.json 2>&1
done
$PY logprob_equivalence.py --diff equiv_base.json equiv_oblit_id.json > oblit_id_diff.json 2>&1
echo PHASE2_ALL_DONE >> run_phase2.log
echo PHASE2B_DONE
