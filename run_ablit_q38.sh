#!/bin/bash
# Qwen3.8-NVFP4 obliteration: build, serve, full eval (chat-endpoint refusal
# on all 3 batteries + base baseline + equivalence + pass@1).
set -x
cd ~/projects/moe-telemetry
export PATH=/usr/local/cuda/bin:$PATH
PY=/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/python
VLLM=/home/tyrel/.venvs/qwen38-nvfp4-vllm/bin/vllm
SRC=/home/tyrel/models/Qwen3.8-27B-NVFP4

wait_health() { for i in $(seq 1 100); do sleep 15; [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8123/health 2>/dev/null)" = "200" ] && return 0; done; return 1; }
kill_server() { pkill -f "bin/vllm serve"; sleep 10; for i in $(seq 1 12); do ss -ltn "sport = :8123" | grep -q 8123 || break; sleep 5; done; }
serve() { setsid nohup $VLLM serve $1 --served-model-name $2 --port 8123 --gpu-memory-utilization 0.40 --max-model-len 8192 --max-num-seqs 48 > srv5_$2.log 2>&1 < /dev/null & wait_health; }

# 1. base refusal baseline on the CHAT path (fair comparison)
serve $SRC base || exit 1
$PY refusal_check.py --chat --url http://localhost:8123 --model base --out refusal_chat_base_v1.json
$PY refusal_check.py --chat --url http://localhost:8123 --model base --prompts refusal_probe_v2.jsonl --out refusal_chat_base_v2.json
$PY refusal_check.py --chat --url http://localhost:8123 --model base --prompts refusal_probe_v3.jsonl --out refusal_chat_base_v3.json
kill_server

# 2. build obliterated checkpoint
if [ ! -f ${SRC}-oblit/obliteration_manifest.json ]; then
  rm -rf ${SRC}-oblit
  $PY obliterate_q38.py $SRC ${SRC}-oblit || { echo BUILD_FAILED; exit 1; }
fi

# 3. eval oblit
serve ${SRC}-oblit oblit || exit 1
$PY refusal_check.py --chat --url http://localhost:8123 --model oblit --out refusal_chat_oblit_v1.json
$PY refusal_check.py --chat --url http://localhost:8123 --model oblit --prompts refusal_probe_v2.jsonl --out refusal_chat_oblit_v2.json
$PY refusal_check.py --chat --url http://localhost:8123 --model oblit --prompts refusal_probe_v3.jsonl --out refusal_chat_oblit_v3.json
$PY logprob_equivalence.py calib_transcripts.jsonl --url http://localhost:8123 --out equiv_q38_oblit_id.json --n 32 > probe_oblit_id.log 2>&1
$PY logprob_equivalence.py ood2_swe.jsonl --url http://localhost:8123 --out equiv_q38_oblit_swe.json --n 32 > probe_oblit_swe.log 2>&1
$PY logprob_equivalence.py ood2_tools.jsonl --url http://localhost:8123 --out equiv_q38_oblit_tools.json --n 32 > probe_oblit_tools.log 2>&1
$PY pass_at_1.py --url http://localhost:8123 --model oblit --n 24 --out pass1_q38_oblit.json
kill_server

# 4. base equivalence + pass@1 for parity diffs
serve $SRC base || exit 1
$PY logprob_equivalence.py calib_transcripts.jsonl --url http://localhost:8123 --out equiv_q38_base_id2.json --n 32 > probe_base_id2.log 2>&1
$PY logprob_equivalence.py ood2_swe.jsonl --url http://localhost:8123 --out equiv_q38_base_swe.json --n 32 > probe_base_swe.log 2>&1
$PY logprob_equivalence.py ood2_tools.jsonl --url http://localhost:8123 --out equiv_q38_base_tools.json --n 32 > probe_base_tools.log 2>&1
$PY pass_at_1.py --url http://localhost:8123 --model base --n 24 --out pass1_q38_base.json
kill_server

# 5. diffs
$PY logprob_equivalence.py --diff equiv_q38_base_id2.json equiv_q38_oblit_id.json > q38_id_diff.json 2>&1
$PY logprob_equivalence.py --diff equiv_q38_base_swe.json equiv_q38_oblit_swe.json > q38_swe_diff.json 2>&1
$PY logprob_equivalence.py --diff equiv_q38_base_tools.json equiv_q38_oblit_tools.json > q38_tools_diff.json 2>&1
echo PHASE_Q38_A_DONE
