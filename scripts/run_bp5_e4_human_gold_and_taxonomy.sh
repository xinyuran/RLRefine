#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${1:?base model path required}"
DPO_ADAPTER="${2:?DPO checkpoint-74 path required}"
E4_ADAPTER="${3:?E4 final checkpoint path required}"
LOG_FILE="${4:-new_plan/logs/bp5_e4_human_gold_and_taxonomy.log}"
FREEZE="reports/model_gates/bp5_e4_research_candidate_freeze.json"
ONE_SHOT_DIR="reports/bp5/human_gold_one_shot"
status=0

mkdir -p "$(dirname "${LOG_FILE}")" "$(dirname "${FREEZE}")" "${ONE_SHOT_DIR}"
python -m unittest discover -v tests 2>&1 | tee "${LOG_FILE}" || status=$?
if [ "${status}" -eq 0 ]; then
  python -m scripts.freeze_bp4_e4_research_candidate \
    --gate reports/bp4_e4_grpo_from_dpo/evaluation/grpo_two_level_gate.json \
    --paired-packet reports/bp4_e4_grpo_from_dpo/evaluation/grpo_paired_evaluation_packet.json \
    --training-artifacts reports/bp4_e4_grpo_from_dpo/grpo_training_artifacts.json \
    --output "${FREEZE}" 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi
if [ "${status}" -eq 0 ]; then
  python -m evaluation.bp5_human_gold_oneshot --base-model "${BASE_MODEL}" --dpo-adapter "${DPO_ADAPTER}" --e4-adapter "${E4_ADAPTER}" --e4-freeze "${FREEZE}" 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi
if [ "${status}" -eq 0 ]; then
  python -m evaluation.schema_error_taxonomy --predictions reports/bp4_e4_grpo_from_dpo/evaluation/dev_predictions.jsonl --output-dir reports/bp5/schema_error_taxonomy/e4_dev --expected-errors 67 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi
if [ "${status}" -eq 0 ]; then
  python -m evaluation.schema_error_taxonomy --predictions reports/bp4_e4_grpo_from_dpo/evaluation/challenge_predictions.jsonl --output-dir reports/bp5/schema_error_taxonomy/e4_challenge --expected-errors 49 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi
if [ "${status}" -eq 0 ]; then
  echo '{"event":"bp5_human_gold_and_taxonomy_runner_complete","status":"PASS","human_gold_used_for_selection":false,"next_execution":"BP5_CONSTRAINED_DECODING_AND_SERVING_BENCHMARK"}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"bp5_human_gold_and_taxonomy_runner_complete\",\"status\":\"FAIL\",\"exit_code\":${status}}" | tee -a "${LOG_FILE}"
fi
exit "${status}"
