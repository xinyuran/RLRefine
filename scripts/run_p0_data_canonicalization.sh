#!/usr/bin/env bash
set -euo pipefail

SFT_FILE="${1:-data/sft_data_with_think_tags_jsonl_output/sft_all_merged.jsonl}"
OUTPUT_DIR="${2:-data/canonical/keyword_v1}"
LOG_FILE="${3:-new_plan/logs/p0_data_canonicalization.log}"
CLEAN_SFT="${OUTPUT_DIR}/sft_clean.jsonl"
CLEAN_GRPO="${OUTPUT_DIR}/grpo_clean.jsonl"

mkdir -p "$(dirname "${LOG_FILE}")"
overall_status=0

echo "{\"event\":\"p0_data_run_start\",\"sft_file\":\"${SFT_FILE}\",\"output_dir\":\"${OUTPUT_DIR}\"}" | tee "${LOG_FILE}"
python --version 2>&1 | tee -a "${LOG_FILE}"
python -c 'from importlib.metadata import version; print("ms-swift", version("ms-swift"))' 2>&1 | tee -a "${LOG_FILE}" || true

python -m unittest -v \
  tests.test_reward_builder \
  tests.test_data_canonicalization 2>&1 | tee -a "${LOG_FILE}" || overall_status=$?

python scripts/canonicalize_sft_data.py \
  --input-file "${SFT_FILE}" \
  --output-dir "${OUTPUT_DIR}" 2>&1 | tee -a "${LOG_FILE}" || overall_status=$?

python rl/convert_sft_to_grpo.py \
  --input_file "${CLEAN_SFT}" \
  --output_file "${CLEAN_GRPO}" 2>&1 | tee -a "${LOG_FILE}" || overall_status=$?

python scripts/validate_p0_reward.py \
  --grpo-file "${CLEAN_GRPO}" \
  --sample-limit 0 2>&1 | tee -a "${LOG_FILE}" || overall_status=$?

if [ "${overall_status}" -eq 0 ]; then
  echo '{"event":"p0_data_run_complete","status":"PASS"}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"p0_data_run_complete\",\"status\":\"FAIL\",\"exit_code\":${overall_status}}" | tee -a "${LOG_FILE}"
fi

exit "${overall_status}"
