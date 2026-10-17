#!/bin/bash
# Phase 2: SWE + tools domains, pass@1 code eval, refusal-ablated checkpoint.
set -x
cd ~/projects/moe-telemetry
PY=/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/python
VLLM=/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/vllm
SRC=/home/tyrel/models/FreeToken-Qwen3.5-35B-A3B-FP8

wait_health() { for i in $(seq 1 80); do sleep 15; [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8123/health 2>/dev/null)" = "200" ] && return 0; done; return 1; }
kill_server() { pkill -f "bin/vllm serve"; sleep 20; }
serve() { setsid nohup $VLLM serve $1 --served-model-name $2 --port 8123 --gpu-memory-utilization 0.35 --max-model-len 8192 > srv2_$2.log 2>&1 < /dev/null & wait_health; }
probe() { $PY logprob_equivalence.py $2 --url http://localhost:8123 --out $3 --n 32 > probe_$3.log 2>&1; }
passat() { $PY pass_at_1.py --url http://localhost:8123 --model $1 --n 24 --out pass1_$1.json; }

[ -f humaneval_local.jsonl ] || $PY make_humaneval.py || exit 1

kill_server
# base: SWE + tools probes + pass@1
serve $SRC base || exit 1
probe base ood2_swe.jsonl equiv_swe_base.json
probe base ood2_tools.jsonl equiv_tools_base.json
passat base
kill_server

for N in 241 224 192 128; do
  serve ${SRC}-pruned$N p$N || exit 1
  probe p$N ood2_swe.jsonl equiv_swe_p$N.json
  probe p$N ood2_tools.jsonl equiv_tools_p$N.json
  [ $N = 241 ] && passat p241
  kill_server
done
echo PHASE2A_DONE

# obliterated checkpoint (CPU build) + eval
if [ ! -d ${SRC}-oblit ]; then
  $PY obliterate.py $SRC ${SRC}-oblit || exit 1
fi
serve ${SRC}-oblit oblit || exit 1
$PY refusal_check.py --url http://localhost:8123 --out refusal_oblit.json
probe oblit calib_transcripts.jsonl equiv_oblit_id.json
probe oblit ood2_swe.jsonl equiv_swe_oblit.json
probe oblit ood2_tools.jsonl equiv_tools_oblit.json
passat oblit
kill_server

# baseline refusal rate for comparison (base already probed; serve once more)
serve $SRC base || exit 1
$PY refusal_check.py --url http://localhost:8123 --out refusal_base.json
kill_server

# diffs
for T in p241 p224 p192 p128 oblit; do
  S=equiv_swe_base.json; [ $T = oblit ] && S=equiv_swe_base.json
  $PY logprob_equivalence.py --diff $S equiv_swe_$T.json > swe_${T}_diff.json 2>&1
  $PY logprob_equivalence.py --diff equiv_tools_base.json equiv_tools_$T.json > tools_${T}_diff.json 2>&1
done
$PY logprob_equivalence.py --diff equiv_base.json equiv_oblit_id.json > oblit_id_diff.json 2>&1
echo PHASE2_ALL_DONE
