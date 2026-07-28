#!/usr/bin/env bash
set -euo pipefail

SFT_FILE="${1:-data/sft_data_with_think_tags_jsonl_output/sft_all_merged.jsonl}"
GRPO_FILE="${2:-data/grpo_jsonl_output/grpo_prompts_p0_with_solution.jsonl}"
LOG_FILE="${3:-new_plan/logs/p0_reward_validation.log}"

mkdir -p "$(dirname "${LOG_FILE}")"

echo "{\"event\":\"p0_server_run_start\",\"sft_file\":\"${SFT_FILE}\",\"grpo_file\":\"${GRPO_FILE}\"}" | tee "${LOG_FILE}"
python --version 2>&1 | tee -a "${LOG_FILE}"
swift --version 2>&1 | tee -a "${LOG_FILE}" || true

overall_status=0

python rl/convert_sft_to_grpo.py \
  --input_file "${SFT_FILE}" \
  --output_file "${GRPO_FILE}" 2>&1 | tee -a "${LOG_FILE}" || overall_status=$?

python -m unittest -v tests.test_reward_builder 2>&1 | tee -a "${LOG_FILE}" || overall_status=$?

python scripts/validate_p0_reward.py \
  --grpo-file "${GRPO_FILE}" \
  --sample-limit 200 2>&1 | tee -a "${LOG_FILE}" || overall_status=$?

if [ "${overall_status}" -eq 0 ]; then
  echo '{"event":"p0_server_run_complete","status":"PASS"}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"p0_server_run_complete\",\"status\":\"FAIL\",\"exit_code\":${overall_status}}" | tee -a "${LOG_FILE}"
fi

exit "${overall_status}"
