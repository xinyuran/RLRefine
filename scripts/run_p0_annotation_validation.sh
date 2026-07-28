#!/usr/bin/env bash
set -euo pipefail

PRIMARY_FILE="${1:-data/canonical/keyword_v1/annotation/v1/annotation_primary.csv}"
SECONDARY_FILE="${2:-data/canonical/keyword_v1/annotation/v1/annotation_secondary.csv}"
TEST_CANDIDATE_FILE="${3:-data/canonical/keyword_v1/splits/sft_test_candidate.jsonl}"
OUTPUT_DIR="${4:-data/canonical/keyword_v1/annotation/v1/review_v1}"
LOG_FILE="${5:-new_plan/logs/p0_annotation_validation.log}"

mkdir -p "$(dirname "${LOG_FILE}")"
overall_status=0

echo "{\"event\":\"p0_annotation_validation_run_start\",\"primary_file\":\"${PRIMARY_FILE}\",\"secondary_file\":\"${SECONDARY_FILE}\",\"test_candidate_file\":\"${TEST_CANDIDATE_FILE}\",\"output_dir\":\"${OUTPUT_DIR}\"}" | tee "${LOG_FILE}"
python --version 2>&1 | tee -a "${LOG_FILE}"

missing_input=0
for input_spec in \
  "primary:${PRIMARY_FILE}" \
  "secondary:${SECONDARY_FILE}" \
  "test_candidate:${TEST_CANDIDATE_FILE}"; do
  input_name="${input_spec%%:*}"
  input_path="${input_spec#*:}"
  if [ -f "${input_path}" ]; then
    input_bytes="$(wc -c < "${input_path}" | tr -d ' ')"
    input_sha256="$(sha256sum "${input_path}" | awk '{print $1}')"
    echo "{\"event\":\"annotation_validation_input\",\"name\":\"${input_name}\",\"path\":\"${input_path}\",\"exists\":true,\"bytes\":${input_bytes},\"sha256\":\"${input_sha256}\"}" | tee -a "${LOG_FILE}"
  else
    echo "{\"event\":\"annotation_validation_input\",\"name\":\"${input_name}\",\"path\":\"${input_path}\",\"exists\":false}" | tee -a "${LOG_FILE}"
    missing_input=1
  fi
done

python -m unittest -v \
  tests.test_reward_builder \
  tests.test_data_canonicalization \
  tests.test_data_split \
  tests.test_annotation_packet \
  tests.test_annotation_validation 2>&1 | tee -a "${LOG_FILE}" || overall_status=$?

if [ "${overall_status}" -eq 0 ] && [ "${missing_input}" -eq 0 ]; then
  python scripts/validate_annotations.py \
    --primary-file "${PRIMARY_FILE}" \
    --secondary-file "${SECONDARY_FILE}" \
    --test-candidate-file "${TEST_CANDIDATE_FILE}" \
    --output-dir "${OUTPUT_DIR}" 2>&1 | tee -a "${LOG_FILE}" || overall_status=$?
else
  if [ "${overall_status}" -ne 0 ]; then
    skip_reason="unit_tests_failed"
  else
    skip_reason="missing_input_file"
    overall_status=1
  fi
  echo "{\"event\":\"annotation_validation_skipped\",\"reason\":\"${skip_reason}\"}" | tee -a "${LOG_FILE}"
fi

if [ "${overall_status}" -eq 0 ]; then
  echo '{"event":"p0_annotation_validation_run_complete","status":"PASS"}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"p0_annotation_validation_run_complete\",\"status\":\"FAIL\",\"exit_code\":${overall_status}}" | tee -a "${LOG_FILE}"
fi

exit "${overall_status}"
