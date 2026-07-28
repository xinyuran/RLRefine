#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-Qwen/Qwen2.5-7B-Instruct}"
LOG_FILE="${2:-new_plan/logs/p1_sft_target_contract_audit.log}"
mkdir -p "$(dirname "${LOG_FILE}")"
status=0

echo '{"event":"p1_sft_target_contract_audit_runner_start","label":"training_data_diagnostic_not_model_evaluation"}' | tee "${LOG_FILE}"

python -m unittest -v tests.test_sft_target_contract 2>&1 | tee -a "${LOG_FILE}" || status=$?
if [ "${status}" -eq 0 ]; then
  python -m scripts.audit_sft_target_contract \
    --model-path "${MODEL_PATH}" 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ] && ! grep -q '"event": "p1_sft_target_contract_audit_complete".*"status": "PASS"' "${LOG_FILE}"; then
  echo '{"event":"p1_sft_target_contract_audit_log_check","status":"FAIL","reason":"missing_audit_pass"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  echo '{"event":"p1_sft_target_contract_audit_runner_complete","status":"PASS"}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"p1_sft_target_contract_audit_runner_complete\",\"status\":\"FAIL\",\"exit_code\":${status}}" | tee -a "${LOG_FILE}"
fi

exit "${status}"
