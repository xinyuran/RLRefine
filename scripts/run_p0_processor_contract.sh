#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="${1:-new_plan/logs/p0_processor_contract.log}"
mkdir -p "$(dirname "${LOG_FILE}")"
status=0

echo '{"event":"p0_processor_contract_start","server_repo":"."}' | tee "${LOG_FILE}"
python --version 2>&1 | tee -a "${LOG_FILE}"

python -m unittest -v \
  tests.test_reward_builder \
  tests.test_data_canonicalization \
  tests.test_data_split \
  tests.test_annotation_packet \
  tests.test_annotation_validation \
  tests.test_gold_freeze \
  tests.test_keyword_schema_contract \
  tests.test_processor_contract 2>&1 | tee -a "${LOG_FILE}" || status=$?

if [ "${status}" -eq 0 ]; then
  python -m scripts.validate_p0_processor_contract 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ] && ! grep -q '"event": "p0_processor_contract_complete".*"status": "PASS"' "${LOG_FILE}"; then
  echo '{"event":"p0_processor_contract_log_check","status":"FAIL","reason":"missing_contract_pass"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  echo '{"event":"p0_processor_contract_runner_complete","status":"PASS"}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"p0_processor_contract_runner_complete\",\"status\":\"FAIL\",\"exit_code\":${status}}" | tee -a "${LOG_FILE}"
fi

exit "${status}"
