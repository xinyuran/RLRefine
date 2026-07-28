#!/usr/bin/env bash
set -euo pipefail

EVALUATION_CONFIG="${1:-configs/experiments/bp4_e4_grpo_from_dpo_evaluation_v1.yaml}"
BP2_CONFIG="${2:-configs/experiments/bp2_e4_grpo_from_dpo_v1.yaml}"
LOG_FILE="${3:-new_plan/logs/bp4_e4_grpo_from_dpo_evaluation.log}"
SKIP_INFERENCE="${BP4_SKIP_E4_GRPO_INFERENCE:-0}"
status=0

mkdir -p "$(dirname "${LOG_FILE}")" reports/bp4_e4_grpo_from_dpo/evaluation
if [ "${SKIP_INFERENCE}" = "1" ]; then
  echo '{"event":"bp4_e4_grpo_evaluation_resume","resume_from":"completed_predictions","training_started":false}' | tee -a "${LOG_FILE}"
else
  echo '{"event":"bp4_e4_grpo_evaluation_start","execution_scope":"frozen_inference_and_gate_only","training_started":false,"human_gold_used":false}' | tee "${LOG_FILE}"
fi

for required in "${EVALUATION_CONFIG}" "${BP2_CONFIG}" \
  reports/bp4_e4_grpo_from_dpo/grpo_training_artifacts.json \
  reports/bp4_reliability_repair/dpo/evaluation/dpo_paired_evaluation_packet.json \
  reports/bp4_reliability_repair/dpo/evaluation/dev_predictions.jsonl \
  reports/bp4_controlled_matrix/base_v2/dev_evaluation.json; do
  if [ ! -f "${required}" ]; then
    echo "{\"event\":\"bp4_e4_grpo_evaluation_input_check\",\"status\":\"FAIL\",\"missing\":\"${required}\"}" | tee -a "${LOG_FILE}"
    status=1
  fi
done
if [ "${status}" -eq 0 ]; then python -m unittest discover -v tests 2>&1 | tee -a "${LOG_FILE}" || status=$?; fi
if [ "${status}" -eq 0 ]; then python -m evaluation.bp4_e4_grpo_from_dpo --evaluation-config "${EVALUATION_CONFIG}" --bp2-config "${BP2_CONFIG}" preflight 2>&1 | tee -a "${LOG_FILE}" || status=$?; fi
if [ "${status}" -eq 0 ] && [ "${SKIP_INFERENCE}" != "1" ]; then python -m evaluation.bp4_e4_grpo_from_dpo --evaluation-config "${EVALUATION_CONFIG}" --bp2-config "${BP2_CONFIG}" infer --batch-size 4 --max-input-tokens 4096 2>&1 | tee -a "${LOG_FILE}" || status=$?; fi
if [ "${status}" -eq 0 ]; then python -m evaluation.bp4_e4_grpo_from_dpo --evaluation-config "${EVALUATION_CONFIG}" --bp2-config "${BP2_CONFIG}" package 2>&1 | tee -a "${LOG_FILE}" || status=$?; fi
if [ "${status}" -eq 0 ]; then echo '{"event":"bp4_e4_grpo_evaluation_runner_complete","status":"PASS","decision":"E4_GRPO_EVALUATION_COMPLETE","training_started":false,"deployment_authorized":false}' | tee -a "${LOG_FILE}"; else echo "{\"event\":\"bp4_e4_grpo_evaluation_runner_complete\",\"status\":\"FAIL\",\"exit_code\":${status},\"training_started\":false}" | tee -a "${LOG_FILE}"; fi
exit "${status}"
