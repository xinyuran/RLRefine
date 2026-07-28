#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${1:-configs/experiments/bp2_dpo_reliability_v1.yaml}"
LOG_FILE="${2:-new_plan/logs/bp4_dpo_reliability_training.log}"
RELIABILITY_CONFIG="${3:-configs/experiments/bp4_reliability_repair_v1.yaml}"
OUTPUT_DIR="reports/training/bp2_dpo_reliability_v1"
ARTIFACT_FILE="reports/bp4_reliability_repair/dpo/dpo_training_artifacts.json"
SKIP_TRAINING="${BP4_SKIP_DPO_TRAINING:-0}"
RESUME_FROM="${BP4_DPO_RESUME_FROM:-}"
status=0

mkdir -p "$(dirname "${LOG_FILE}")" "$(dirname "${ARTIFACT_FILE}")"
if [ "${SKIP_TRAINING}" = "1" ]; then
  echo '{"event":"bp4_dpo_runner_resume","work_package":"E2_DPO_RELIABILITY","resume_from":"completed_training_artifacts","authorization_scope":"DPO_RELIABILITY_REPAIR_ONLY","grpo_authorized":false}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"bp4_dpo_runner_start\",\"work_package\":\"E2_DPO_RELIABILITY\",\"authorization_scope\":\"DPO_RELIABILITY_REPAIR_ONLY\",\"resume_checkpoint\":\"${RESUME_FROM}\",\"grpo_authorized\":false}" | tee "${LOG_FILE}"
fi
python --version 2>&1 | tee -a "${LOG_FILE}"

if [ ! -f "${CONFIG_FILE}" ] || [ ! -f "${RELIABILITY_CONFIG}" ]; then
  echo '{"event":"bp4_dpo_input_check","status":"FAIL","reason":"missing_config"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  python -m scripts.check_ms_swift_dpo_compat \
    2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m unittest discover -v tests 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m evaluation.bp4_reliability_gate \
    --config "${RELIABILITY_CONFIG}" \
    2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m scripts.bp2_pipeline launch \
    --config "${CONFIG_FILE}" --stage dpo --dry-run --require-authorized \
    2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ] && [ "${SKIP_TRAINING}" != "1" ]; then
  if [ -n "${RESUME_FROM}" ]; then
    python -m scripts.bp2_pipeline launch \
      --config "${CONFIG_FILE}" --stage dpo --resume-from "${RESUME_FROM}" \
      2>&1 | tee -a "${LOG_FILE}" || status=$?
  else
    python -m scripts.bp2_pipeline launch \
      --config "${CONFIG_FILE}" --stage dpo \
      2>&1 | tee -a "${LOG_FILE}" || status=$?
  fi
fi

if [ "${status}" -eq 0 ] && [ "${SKIP_TRAINING}" = "1" ]; then
  echo '{"event":"bp4_dpo_training_reuse","status":"PASS","reason":"training_already_completed_no_retraining"}' | tee -a "${LOG_FILE}"
fi

if [ "${status}" -eq 0 ]; then
  python -m scripts.bp2_pipeline inspect-artifacts \
    --output-dir "${OUTPUT_DIR}" --report-file "${ARTIFACT_FILE}" \
    2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -c 'import json; from pathlib import Path; p=Path("reports/bp4_reliability_repair/dpo/dpo_training_artifacts.json"); x=json.loads(p.read_text(encoding="utf-8")); assert x["checkpoint_count"]>0; assert x["best_model_checkpoint"]; print(json.dumps({"event":"bp4_dpo_artifact_check","status":"PASS","checkpoint_count":x["checkpoint_count"],"best_model_checkpoint":x["best_model_checkpoint"]},sort_keys=True))' 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  echo '{"event":"bp4_dpo_runner_complete","status":"PASS","decision":"DPO_TRAINING_COMPLETE_EVALUATION_PENDING","training_started":true,"training_completed":true,"grpo_authorized":false}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"bp4_dpo_runner_complete\",\"status\":\"FAIL\",\"exit_code\":${status},\"grpo_authorized\":false}" | tee -a "${LOG_FILE}"
fi

exit "${status}"
