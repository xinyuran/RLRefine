#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${1:-configs/experiments/bp2_post_training_pipeline_v1.yaml}"
OUTPUT_DIR="${2:-reports/bp2_post_training_pipeline}"
LOG_FILE="${3:-new_plan/logs/bp2_post_training_pipeline.log}"
status=0

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")"
echo "{\"event\":\"bp2_runner_start\",\"work_package\":\"BP2_POST_TRAINING_PIPELINE\",\"config\":\"${CONFIG_FILE}\",\"output_dir\":\"${OUTPUT_DIR}\",\"execution_policy\":\"engineering_only_no_training\"}" | tee "${LOG_FILE}"
python --version 2>&1 | tee -a "${LOG_FILE}"

if [ ! -f "${CONFIG_FILE}" ]; then
  echo '{"event":"bp2_input_check","status":"FAIL","reason":"missing_config"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  python -m unittest discover -v tests 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  for stage in sft dpo grpo; do
    python -m scripts.bp2_pipeline launch \
      --config "${CONFIG_FILE}" \
      --stage "${stage}" \
      --dry-run 2>&1 | tee -a "${LOG_FILE}" || status=$?
    if [ "${status}" -ne 0 ]; then
      break
    fi
  done
fi

if [ "${status}" -eq 0 ]; then
  python -m scripts.bp2_pipeline prepare \
    --config "${CONFIG_FILE}" \
    --output-dir "${OUTPUT_DIR}" 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ] && ! grep -q '"event": "bp2_gate_complete".*"status": "PASS"' "${LOG_FILE}"; then
  echo '{"event":"bp2_log_check","status":"FAIL","reason":"missing_gate_pass"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  echo '{"event":"bp2_runner_complete","status":"PASS","decision":"ACCEPT_BP2_ENGINEERING","next_work_package":"BP3_REWARD_V2_AND_PREFERENCE_DATA","training_started":false}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"bp2_runner_complete\",\"status\":\"FAIL\",\"exit_code\":${status},\"training_started\":false}" | tee -a "${LOG_FILE}"
fi

exit "${status}"
