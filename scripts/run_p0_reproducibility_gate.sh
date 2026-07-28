#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${1:-configs/experiments/p1_sft_lora_v1.yaml}"
OUTPUT_DIR="${2:-reports/lineage/p1_sft_lora_v1}"
LOG_FILE="${3:-new_plan/logs/p0_reproducibility_gate.log}"
mkdir -p "$(dirname "${LOG_FILE}")"
status=0

echo "{\"event\":\"p0_reproducibility_gate_start\",\"config_file\":\"${CONFIG_FILE}\",\"output_dir\":\"${OUTPUT_DIR}\",\"server_repo\":\".\"}" | tee "${LOG_FILE}"
python --version 2>&1 | tee -a "${LOG_FILE}"

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "{\"event\":\"p0_reproducibility_gate_input\",\"status\":\"FAIL\",\"reason\":\"missing_config\",\"path\":\"${CONFIG_FILE}\"}" | tee -a "${LOG_FILE}"
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
    tests.test_keyword_evaluator \
    tests.test_baseline_inference \
    tests.test_baseline_diagnostics \
    tests.test_prompt_optimization_diagnostics \
    tests.test_experiment_manifest \
    tests.test_sft_launch \
    tests.test_swift_yaml_launcher 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m scripts.prepare_experiment_manifest \
    --config "${CONFIG_FILE}" \
    --output-dir "${OUTPUT_DIR}" 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m scripts.swift_yaml_launcher \
    --config "${OUTPUT_DIR}/swift_sft_config.yaml" \
    --dry-run 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ] && ! grep -q '"event": "p0_reproducibility_manifest_complete".*"status": "PASS"' "${LOG_FILE}"; then
  echo '{"event":"p0_reproducibility_gate_log_check","status":"FAIL","reason":"missing_manifest_pass"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  echo '{"event":"p0_reproducibility_gate_complete","status":"PASS"}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"p0_reproducibility_gate_complete\",\"status\":\"FAIL\",\"exit_code\":${status}}" | tee -a "${LOG_FILE}"
fi

exit "${status}"
