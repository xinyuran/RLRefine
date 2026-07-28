#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${1:-configs/experiments/bp4_controlled_model_matrix_v1.yaml}"
BP2_CONFIG="${2:-configs/experiments/bp2_post_training_pipeline_v1.yaml}"
LOG_FILE="${3:-new_plan/logs/bp4_controlled_sft.log}"
ARTIFACT_FILE="reports/bp4_controlled_matrix/sft/sft_training_artifacts.json"
SKIP_TRAINING="${BP4_SKIP_SFT_TRAINING:-0}"
status=0

mkdir -p "$(dirname "${LOG_FILE}")" "$(dirname "${ARTIFACT_FILE}")"
if [ "${SKIP_TRAINING}" = "1" ]; then
  echo '{"event":"bp4_sft_runner_resume","work_package":"BP4_CONTROLLED_SFT","resume_from":"completed_training_artifacts","authorization_scope":"SFT_ONLY","dpo_authorized":false,"grpo_authorized":false}' | tee -a "${LOG_FILE}"
else
  echo '{"event":"bp4_sft_runner_start","work_package":"BP4_CONTROLLED_SFT","authorization_scope":"SFT_ONLY","dpo_authorized":false,"grpo_authorized":false}' | tee "${LOG_FILE}"
fi
python --version 2>&1 | tee -a "${LOG_FILE}"

if [ ! -f "${CONFIG_FILE}" ] || [ ! -f "${BP2_CONFIG}" ]; then
  echo '{"event":"bp4_sft_input_check","status":"FAIL","reason":"missing_config"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  python -m unittest discover -v tests 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m evaluation.bp4_sft_v2 \
    --config "${CONFIG_FILE}" preflight \
    2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ] && [ "${SKIP_TRAINING}" != "1" ]; then
  python -m scripts.bp2_pipeline launch \
    --config "${BP2_CONFIG}" --stage sft \
    2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ] && [ "${SKIP_TRAINING}" = "1" ]; then
  echo '{"event":"bp4_sft_training_reuse","status":"PASS","reason":"training_already_completed_no_retraining"}' | tee -a "${LOG_FILE}"
fi

if [ "${status}" -eq 0 ]; then
  python -m scripts.bp2_pipeline inspect-artifacts \
    --output-dir reports/training/bp2_sft_v2 \
    --report-file "${ARTIFACT_FILE}" \
    2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m evaluation.bp4_sft_v2 \
    --config "${CONFIG_FILE}" --artifacts "${ARTIFACT_FILE}" \
    infer --batch-size 4 --max-input-tokens 4096 \
    2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m evaluation.bp4_sft_v2 \
    --config "${CONFIG_FILE}" --artifacts "${ARTIFACT_FILE}" \
    package 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -c 'import json; from pathlib import Path; p=Path("reports/bp4_controlled_matrix/sft"); gate=json.loads((p/"sft_gate.json").read_text(encoding="utf-8")); feedback=json.loads((p/"sft_feedback.json").read_text(encoding="utf-8")); assert gate["status"]=="PASS"; assert gate["decision"] in {"ACCEPT_SFT_CANDIDATE","REJECT_SFT_CANDIDATE"}; assert gate["downstream_training_authorized"] is False; assert gate["dpo_started"] is False and gate["grpo_started"] is False; assert len(feedback["return_files"])==13; print(json.dumps({"event":"bp4_sft_independent_check","status":"PASS","decision":gate["decision"],"downstream_training_authorized":False,"return_file_count":13},sort_keys=True))' 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  decision="$(python -c 'import json; print(json.load(open("reports/bp4_controlled_matrix/sft/sft_gate.json",encoding="utf-8"))["decision"])')"
  echo "{\"event\":\"bp4_sft_runner_complete\",\"status\":\"PASS\",\"decision\":\"${decision}\",\"downstream_training_authorized\":false,\"dpo_started\":false,\"grpo_started\":false}" | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"bp4_sft_runner_complete\",\"status\":\"FAIL\",\"exit_code\":${status},\"downstream_training_authorized\":false,\"dpo_started\":false,\"grpo_started\":false}" | tee -a "${LOG_FILE}"
fi

exit "${status}"
