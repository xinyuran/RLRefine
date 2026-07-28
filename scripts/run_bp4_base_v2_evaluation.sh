#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${1:-configs/experiments/bp4_controlled_model_matrix_v1.yaml}"
TASK_FILE="${2:-reports/bp4_controlled_matrix/base_v2_evaluation_task.json}"
LOG_FILE="${3:-new_plan/logs/bp4_base_v2_evaluation.log}"
status=0

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
mkdir -p "$(dirname "${LOG_FILE}")"
echo "{\"event\":\"bp4_base_runner_start\",\"work_package\":\"BP4_BASE_V2_EVALUATION\",\"config\":\"${CONFIG_FILE}\",\"task\":\"${TASK_FILE}\",\"execution_policy\":\"base_inference_only_no_training\"}" | tee "${LOG_FILE}"
python --version 2>&1 | tee -a "${LOG_FILE}"

if [ ! -f "${CONFIG_FILE}" ] || [ ! -f "${TASK_FILE}" ]; then
  echo '{"event":"bp4_base_input_check","status":"FAIL","reason":"missing_config_or_task"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  python -m unittest discover -v tests 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m evaluation.bp4_base_v2 \
    --config "${CONFIG_FILE}" \
    --task "${TASK_FILE}" \
    infer --batch-size 4 --max-input-tokens 4096 \
    2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m evaluation.bp4_base_v2 \
    --config "${CONFIG_FILE}" \
    --task "${TASK_FILE}" \
    package 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m scripts.bp4_matrix \
    --config "${CONFIG_FILE}" \
    sft-authorization-candidate \
    --base-packet reports/bp4_controlled_matrix/base_v2/base_evaluation_packet.json \
    --output reports/bp4_controlled_matrix/authorization_candidates/bp2_sft_ready.json \
    2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -c 'import json; from pathlib import Path; base=Path("reports/bp4_controlled_matrix/base_v2"); gate=json.loads((base/"base_gate.json").read_text(encoding="utf-8")); packet=json.loads((base/"base_evaluation_packet.json").read_text(encoding="utf-8")); candidate=json.loads(Path("reports/bp4_controlled_matrix/authorization_candidates/bp2_sft_ready.json").read_text(encoding="utf-8")); assert gate["status"] == "PASS" and gate["decision"] == "ACCEPT_BP4_BASE_V2"; assert packet["status"] == "PASS" and packet["dev_rows"] == 331 and packet["challenge_rows"] == 140; assert packet["human_gold_used"] is False and packet["training_started"] is False; assert candidate["activation_status"] == "CANDIDATE_NOT_ACTIVE" and candidate["human_approval_required"] is True; assert not Path("reports/authorizations/bp2_sft.json").exists(); print(json.dumps({"event":"bp4_base_independent_check","status":"PASS","base_v2":"ACCEPTED","sft_authorization":"CANDIDATE_NOT_ACTIVE","training_started":False},sort_keys=True))' 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  echo '{"event":"bp4_base_runner_complete","status":"PASS","decision":"ACCEPT_BP4_BASE_V2","next_execution":"BP4_SFT_AUTHORIZATION_REVIEW","sft_authorization":"CANDIDATE_NOT_ACTIVE","training_started":false}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"bp4_base_runner_complete\",\"status\":\"FAIL\",\"exit_code\":${status},\"training_started\":false}" | tee -a "${LOG_FILE}"
fi

exit "${status}"
