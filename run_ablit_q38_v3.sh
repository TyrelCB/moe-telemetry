#!/bin/bash
# Arditi v2: two-sided per-layer attention ablation, real refusal activations.
set -x
cd ~/projects/moe-telemetry
export PATH=/usr/local/cuda/bin:$PATH
PY=/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/python
VLLM=/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/vllm
SRC=/home/tyrel/models/Qwen3.8-27B-NVFP4
DST=${SRC}-arditi3
wait_health() { for i in $(seq 1 100); do sleep 15; [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8123/health 2>/dev/null)" = "200" ] && return 0; done; return 1; }
kill_server() { pkill -f "bin/vllm serve"; sleep 10; }
serve() { setsid nohup $VLLM serve $1 --served-model-name $2 --port 8123 --gpu-memory-utilization 0.40 --max-model-len 8192 --max-num-seqs 48 > srv6_$2.log 2>&1 < /dev/null & wait_health; }

rm -rf $DST
$PY obliterate_q38_v2.py $SRC $DST --dirs arditi_dirs.pt --factor 1.0 --sides write || { echo BUILD_FAILED; exit 1; }
echo BUILD_DONE

serve $DST oblit3 || { echo SERVE_FAILED; exit 1; }
$PY refusal_check.py --chat --url http://localhost:8123 --model oblit3 --out refusal_chat_oblit3_v1.json
$PY refusal_check.py --chat --url http://localhost:8123 --model oblit3 --prompts refusal_probe_v2.jsonl --out refusal_chat_oblit3_v2.json
$PY refusal_check.py --chat --url http://localhost:8123 --model oblit3 --prompts refusal_probe_v3.jsonl --out refusal_chat_oblit3_v3.json
$PY refusal_check.py --chat --url http://localhost:8123 --model oblit3 --prompts orbench_prompts.jsonl --out refusal_orbench_oblit3.json
$PY logprob_equivalence.py calib_transcripts.jsonl --url http://localhost:8123 --out equiv_q38_oblit3_id.json --n 32 > probe_oblit3_id.log 2>&1
$PY logprob_equivalence.py ood2_swe.jsonl --url http://localhost:8123 --out equiv_q38_oblit3_swe.json --n 32 > probe_oblit3_swe.log 2>&1
$PY logprob_equivalence.py ood2_tools.jsonl --url http://localhost:8123 --out equiv_q38_oblit3_tools.json --n 32 > probe_oblit3_tools.log 2>&1
$PY pass_at_1.py --url http://localhost:8123 --model oblit3 --n 24 --out pass1_q38_oblit3.json
kill_server

$PY logprob_equivalence.py --diff equiv_q38_base_id2.json equiv_q38_oblit3_id.json > q38_id_diff3.json 2>&1
$PY logprob_equivalence.py --diff equiv_q38_base_swe.json equiv_q38_oblit3_swe.json > q38_swe_diff3.json 2>&1
$PY logprob_equivalence.py --diff equiv_q38_base_tools.json equiv_q38_oblit3_tools.json > q38_tools_diff3.json 2>&1
echo PHASE_Q38_ARDITI3_DONE
