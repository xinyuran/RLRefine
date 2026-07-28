#!/usr/bin/env bash
set -euo pipefail

RELIABILITY_CONFIG="${1:-configs/experiments/bp4_reliability_repair_v1.yaml}"
MATRIX_CONFIG="${2:-configs/experiments/bp4_controlled_model_matrix_v1.yaml}"
BP2_CONFIG="${3:-configs/experiments/bp2_dpo_reliability_v1.yaml}"
LOG_FILE="${4:-new_plan/logs/bp4_dpo_reliability_evaluation.log}"
ARTIFACT_FILE="reports/bp4_reliability_repair/dpo/dpo_training_artifacts.json"
OUTPUT_DIR="reports/bp4_reliability_repair/dpo/evaluation"
SKIP_INFERENCE="${BP4_SKIP_DPO_INFERENCE:-0}"
status=0

mkdir -p "$(dirname "${LOG_FILE}")" "${OUTPUT_DIR}"
if [ "${SKIP_INFERENCE}" = "1" ]; then
  echo '{"event":"bp4_dpo_evaluation_resume","work_package":"E2_DPO_RELIABILITY_EVALUATION","resume_from":"completed_predictions","training_started":false,"grpo_authorized":false}' | tee -a "${LOG_FILE}"
else
  echo '{"event":"bp4_dpo_evaluation_start","work_package":"E2_DPO_RELIABILITY_EVALUATION","execution_scope":"inference_and_two_level_gate_only","training_started":false,"grpo_authorized":false}' | tee "${LOG_FILE}"
fi
python --version 2>&1 | tee -a "${LOG_FILE}"

for required in \
  "${RELIABILITY_CONFIG}" \
  "${MATRIX_CONFIG}" \
  "${BP2_CONFIG}" \
  "${ARTIFACT_FILE}" \
  "reports/bp4_controlled_matrix/sft/dev_predictions.jsonl"; do
  if [ ! -f "${required}" ]; then
    echo "{\"event\":\"bp4_dpo_evaluation_input_check\",\"status\":\"FAIL\",\"missing\":\"${required}\"}" | tee -a "${LOG_FILE}"
    status=1
  fi
done

if [ "${status}" -eq 0 ]; then
  python -m unittest discover -v tests 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m evaluation.bp4_dpo_reliability \
    --reliability-config "${RELIABILITY_CONFIG}" \
    --matrix-config "${MATRIX_CONFIG}" \
    --bp2-config "${BP2_CONFIG}" \
    --artifacts "${ARTIFACT_FILE}" \
    preflight 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ] && [ "${SKIP_INFERENCE}" != "1" ]; then
  python -m evaluation.bp4_dpo_reliability \
    --reliability-config "${RELIABILITY_CONFIG}" \
    --matrix-config "${MATRIX_CONFIG}" \
    --bp2-config "${BP2_CONFIG}" \
    --artifacts "${ARTIFACT_FILE}" \
    infer --batch-size 4 --max-input-tokens 4096 \
    2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ] && [ "${SKIP_INFERENCE}" = "1" ]; then
  echo '{"event":"bp4_dpo_inference_reuse","status":"PASS","reason":"predictions_already_completed_no_reinference"}' | tee -a "${LOG_FILE}"
fi

if [ "${status}" -eq 0 ]; then
  python -m evaluation.bp4_dpo_reliability \
    --reliability-config "${RELIABILITY_CONFIG}" \
    --matrix-config "${MATRIX_CONFIG}" \
    --bp2-config "${BP2_CONFIG}" \
    --artifacts "${ARTIFACT_FILE}" \
    package 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -c 'import json; from pathlib import Path; p=Path("reports/bp4_reliability_repair/dpo/evaluation"); gate=json.loads((p/"dpo_two_level_gate.json").read_text(encoding="utf-8")); feedback=json.loads((p/"dpo_evaluation_feedback.json").read_text(encoding="utf-8")); assert gate["status"]=="PASS"; assert gate["research_progression_decision"] in {"ACCEPT_DPO_RESEARCH_PROGRESSION","REJECT_DPO_RESEARCH_PROGRESSION"}; assert gate["deployment_decision"] in {"ACCEPT_DPO_DEPLOYMENT","REJECT_DPO_DEPLOYMENT"}; assert gate["grpo_training_authorized"] is False; assert len(feedback["return_files"])>=10; print(json.dumps({"event":"bp4_dpo_evaluation_independent_check","status":"PASS","research_progression_decision":gate["research_progression_decision"],"deployment_decision":gate["deployment_decision"],"grpo_training_authorized":False,"return_file_count":len(feedback["return_files"])},sort_keys=True))' 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  echo '{"event":"bp4_dpo_evaluation_runner_complete","status":"PASS","decision":"DPO_EVALUATION_COMPLETE","training_started":false,"grpo_authorized":false}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"bp4_dpo_evaluation_runner_complete\",\"status\":\"FAIL\",\"exit_code\":${status},\"training_started\":false,\"grpo_authorized\":false}" | tee -a "${LOG_FILE}"
fi

exit "${status}"
