#!/usr/bin/env bash
set -euo pipefail

TEST_CANDIDATE="${1:-data/canonical/keyword_v1/splits/sft_test_candidate.jsonl}"
OUTPUT_DIR="${2:-data/canonical/keyword_v1/annotation/v1}"
LOG_FILE="${3:-new_plan/logs/p0_annotation_packet.log}"

mkdir -p "$(dirname "${LOG_FILE}")"
overall_status=0

echo "{\"event\":\"p0_annotation_run_start\",\"test_candidate\":\"${TEST_CANDIDATE}\",\"output_dir\":\"${OUTPUT_DIR}\"}" | tee "${LOG_FILE}"
python --version 2>&1 | tee -a "${LOG_FILE}"

python -m unittest -v \
  tests.test_reward_builder \
  tests.test_data_canonicalization \
  tests.test_data_split \
  tests.test_annotation_packet 2>&1 | tee -a "${LOG_FILE}" || overall_status=$?

python scripts/prepare_annotation_packet.py \
  --input-file "${TEST_CANDIDATE}" \
  --output-dir "${OUTPUT_DIR}" \
  --secondary-ratio 0.20 2>&1 | tee -a "${LOG_FILE}" || overall_status=$?

if [ "${overall_status}" -eq 0 ]; then
  echo '{"event":"p0_annotation_run_complete","status":"PASS"}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"p0_annotation_run_complete\",\"status\":\"FAIL\",\"exit_code\":${overall_status}}" | tee -a "${LOG_FILE}"
fi

exit "${overall_status}"
