#!/usr/bin/env bash
set -euo pipefail

CLEAN_SFT="${1:-data/canonical/keyword_v1/sft_clean.jsonl}"
OUTPUT_DIR="${2:-data/canonical/keyword_v1/splits}"
LOG_FILE="${3:-new_plan/logs/p0_group_split.log}"

mkdir -p "$(dirname "${LOG_FILE}")"
overall_status=0

echo "{\"event\":\"p0_split_run_start\",\"clean_sft\":\"${CLEAN_SFT}\",\"output_dir\":\"${OUTPUT_DIR}\"}" | tee "${LOG_FILE}"
python --version 2>&1 | tee -a "${LOG_FILE}"
python -c 'from importlib.metadata import version; print("ms-swift", version("ms-swift"))' 2>&1 | tee -a "${LOG_FILE}" || true

python -m unittest -v \
  tests.test_reward_builder \
  tests.test_data_canonicalization \
  tests.test_data_split 2>&1 | tee -a "${LOG_FILE}" || overall_status=$?

python scripts/split_canonical_data.py \
  --input-file "${CLEAN_SFT}" \
  --output-dir "${OUTPUT_DIR}" \
  --train-ratio 0.75 \
  --dev-ratio 0.10 \
  --test-ratio 0.15 2>&1 | tee -a "${LOG_FILE}" || overall_status=$?

for split_name in train dev test_candidate; do
  python rl/convert_sft_to_grpo.py \
    --input_file "${OUTPUT_DIR}/sft_${split_name}.jsonl" \
    --output_file "${OUTPUT_DIR}/grpo_${split_name}.jsonl" 2>&1 | tee -a "${LOG_FILE}" || overall_status=$?

  python scripts/validate_p0_reward.py \
    --grpo-file "${OUTPUT_DIR}/grpo_${split_name}.jsonl" \
    --sample-limit 0 2>&1 | tee -a "${LOG_FILE}" || overall_status=$?
done

if [ "${overall_status}" -eq 0 ]; then
  echo '{"event":"p0_split_run_complete","status":"PASS"}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"p0_split_run_complete\",\"status\":\"FAIL\",\"exit_code\":${overall_status}}" | tee -a "${LOG_FILE}"
fi

exit "${overall_status}"
