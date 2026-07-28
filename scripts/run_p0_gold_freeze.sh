#!/usr/bin/env bash
set -euo pipefail

PRIMARY_FILE="${1:-data/canonical/keyword_v1/annotation/v1/annotation_primary.csv}"
SECONDARY_FILE="${2:-data/canonical/keyword_v1/annotation/v1/annotation_secondary_v2_annotated.csv}"
DISAGREEMENT_FILE="${3:-data/canonical/keyword_v1/annotation/v1/review_v2/annotation_disagreements.csv}"
ADJUDICATED_FILE="${4:-data/canonical/keyword_v1/annotation/v1/review_v2/annotation_disagreements_adjudicated.csv}"
TEST_CANDIDATE_FILE="${5:-data/canonical/keyword_v1/splits/sft_test_candidate.jsonl}"
OUTPUT_DIR="${6:-data/canonical/keyword_v1/gold/v1}"
LOG_FILE="${7:-new_plan/logs/p0_gold_freeze.log}"

mkdir -p "$(dirname "${LOG_FILE}")"
overall_status=0
missing_input=0

echo "{\"event\":\"p0_gold_freeze_run_start\",\"output_dir\":\"${OUTPUT_DIR}\"}" | tee "${LOG_FILE}"
python --version 2>&1 | tee -a "${LOG_FILE}"

for input_spec in \
  "primary:${PRIMARY_FILE}" \
  "secondary:${SECONDARY_FILE}" \
  "disagreements:${DISAGREEMENT_FILE}" \
  "adjudicated:${ADJUDICATED_FILE}" \
  "test_candidate:${TEST_CANDIDATE_FILE}"; do
  input_name="${input_spec%%:*}"
  input_path="${input_spec#*:}"
  if [ -f "${input_path}" ]; then
    input_bytes="$(wc -c < "${input_path}" | tr -d ' ')"
    input_sha256="$(sha256sum "${input_path}" | awk '{print $1}')"
    echo "{\"event\":\"gold_freeze_input\",\"name\":\"${input_name}\",\"path\":\"${input_path}\",\"exists\":true,\"bytes\":${input_bytes},\"sha256\":\"${input_sha256}\"}" | tee -a "${LOG_FILE}"
  else
    echo "{\"event\":\"gold_freeze_input\",\"name\":\"${input_name}\",\"path\":\"${input_path}\",\"exists\":false}" | tee -a "${LOG_FILE}"
    missing_input=1
  fi
done

python -m unittest -v \
  tests.test_reward_builder \
  tests.test_data_canonicalization \
  tests.test_data_split \
  tests.test_annotation_packet \
  tests.test_annotation_validation \
  tests.test_gold_freeze 2>&1 | tee -a "${LOG_FILE}" || overall_status=$?

if [ "${overall_status}" -eq 0 ] && [ "${missing_input}" -eq 0 ]; then
  python scripts/freeze_gold_test.py \
    --primary-file "${PRIMARY_FILE}" \
    --secondary-file "${SECONDARY_FILE}" \
    --disagreement-file "${DISAGREEMENT_FILE}" \
    --adjudicated-file "${ADJUDICATED_FILE}" \
    --test-candidate-file "${TEST_CANDIDATE_FILE}" \
    --output-dir "${OUTPUT_DIR}" 2>&1 | tee -a "${LOG_FILE}" || overall_status=$?
else
  if [ "${overall_status}" -ne 0 ]; then
    skip_reason="unit_tests_failed"
  else
    skip_reason="missing_input_file"
    overall_status=1
  fi
  echo "{\"event\":\"gold_freeze_skipped\",\"reason\":\"${skip_reason}\"}" | tee -a "${LOG_FILE}"
fi

if [ "${overall_status}" -eq 0 ]; then
  echo '{"event":"p0_gold_freeze_run_complete","status":"PASS"}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"p0_gold_freeze_run_complete\",\"status\":\"FAIL\",\"exit_code\":${overall_status}}" | tee -a "${LOG_FILE}"
fi

exit "${overall_status}"
