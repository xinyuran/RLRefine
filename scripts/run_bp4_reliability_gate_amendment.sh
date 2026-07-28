#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${1:-configs/experiments/bp4_reliability_repair_v1.yaml}"
LOG_FILE="${2:-new_plan/logs/bp4_reliability_gate_amendment.log}"
status=0

mkdir -p "$(dirname "${LOG_FILE}")"
echo '{"event":"bp4_reliability_runner_start","work_package":"BP4_TWO_LEVEL_GATE_AND_DPO_RELIABILITY_REPAIR","execution_scope":"gate_amendment_and_inactive_candidate_only","dpo_training_authorized":false,"grpo_training_authorized":false}' | tee "${LOG_FILE}"

python -m unittest discover -v tests 2>&1 | tee -a "${LOG_FILE}" || status=$?

if [ "${status}" -eq 0 ]; then
  python -m evaluation.bp4_reliability_gate \
    --config "${CONFIG_FILE}" 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ] && ! grep -q '"decision": "ACCEPT_BP4_TWO_LEVEL_GATE_AMENDMENT"' "${LOG_FILE}"; then
  echo '{"event":"bp4_reliability_log_check","status":"FAIL","reason":"missing_amendment_acceptance"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  echo '{"event":"bp4_reliability_runner_complete","status":"PASS","decision":"ACCEPT_BP4_TWO_LEVEL_GATE_AMENDMENT","next_execution":"HUMAN_REVIEW_DPO_RELIABILITY_AUTHORIZATION","dpo_training_authorized":false,"grpo_training_authorized":false,"training_started":false}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"bp4_reliability_runner_complete\",\"status\":\"FAIL\",\"exit_code\":${status},\"training_started\":false}" | tee -a "${LOG_FILE}"
fi

exit "${status}"
