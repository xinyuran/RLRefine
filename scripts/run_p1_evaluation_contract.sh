#!/usr/bin/env bash
set -euo pipefail

GOLD_FILE="${1:-data/canonical/keyword_v1/gold/v1/gold_test.jsonl}"
OUTPUT_DIR="${2:-reports/evaluation_contract/v1}"
LOG_FILE="${3:-new_plan/logs/p1_evaluation_contract.log}"
mkdir -p "$(dirname "${LOG_FILE}")"
status=0

echo "{\"event\":\"p1_evaluation_contract_start\",\"gold_file\":\"${GOLD_FILE}\",\"output_dir\":\"${OUTPUT_DIR}\",\"server_repo\":\".\"}" | tee "${LOG_FILE}"
python --version 2>&1 | tee -a "${LOG_FILE}"

if [ ! -f "${GOLD_FILE}" ]; then
  echo "{\"event\":\"p1_evaluation_contract_input\",\"status\":\"FAIL\",\"reason\":\"missing_gold_file\",\"path\":\"${GOLD_FILE}\"}" | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  python -m unittest -v \
    tests.test_reward_builder \
    tests.test_data_canonicalization \
    tests.test_data_split \
    tests.test_annotation_packet \
    tests.test_annotation_validation \
    tests.test_gold_freeze \
    tests.test_keyword_schema_contract \
    tests.test_processor_contract \
    tests.test_keyword_evaluator 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m scripts.validate_p1_evaluation_engine \
    --gold-file "${GOLD_FILE}" \
    --output-dir "${OUTPUT_DIR}" 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ] && ! grep -q '"event": "p1_evaluation_contract_complete".*"status": "PASS"' "${LOG_FILE}"; then
  echo '{"event":"p1_evaluation_contract_log_check","status":"FAIL","reason":"missing_contract_pass"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  echo '{"event":"p1_evaluation_contract_runner_complete","status":"PASS"}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"p1_evaluation_contract_runner_complete\",\"status\":\"FAIL\",\"exit_code\":${status}}" | tee -a "${LOG_FILE}"
fi

exit "${status}"
