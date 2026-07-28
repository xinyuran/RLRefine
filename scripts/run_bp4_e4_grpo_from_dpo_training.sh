#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${1:-configs/experiments/bp2_e4_grpo_from_dpo_v1.yaml}"
LOG_FILE="${2:-new_plan/logs/bp4_e4_grpo_from_dpo_training.log}"
OUTPUT_DIR="reports/training/bp2_e4_grpo_from_dpo_v1"
ARTIFACT_FILE="reports/bp4_e4_grpo_from_dpo/grpo_training_artifacts.json"
SKIP_TRAINING="${BP4_SKIP_E4_GRPO_TRAINING:-0}"
RESUME_FROM="${BP4_E4_GRPO_RESUME_FROM:-}"
status=0

mkdir -p "$(dirname "${LOG_FILE}")" "$(dirname "${ARTIFACT_FILE}")"
if [ "${SKIP_TRAINING}" = "1" ]; then
  echo '{"event":"bp4_e4_grpo_runner_resume","matrix_id":"E4_GRPO_FROM_DPO","resume_from":"completed_training_artifacts","authorization_scope":"E4_GRPO_FROM_DPO_ONLY"}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"bp4_e4_grpo_runner_start\",\"matrix_id\":\"E4_GRPO_FROM_DPO\",\"authorization_scope\":\"E4_GRPO_FROM_DPO_ONLY\",\"resume_checkpoint\":\"${RESUME_FROM}\",\"human_gold_used\":false}" | tee "${LOG_FILE}"
fi

python --version 2>&1 | tee -a "${LOG_FILE}"
python -c "import importlib.metadata as m; print('ms-swift', m.version('ms-swift')); print('torch', m.version('torch')); print('transformers', m.version('transformers'))" 2>&1 | tee -a "${LOG_FILE}" || status=$?
nvidia-smi -i 7 --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader 2>&1 | tee -a "${LOG_FILE}" || status=$?

if [ ! -f "${CONFIG_FILE}" ]; then
  echo '{"event":"bp4_e4_grpo_input_check","status":"FAIL","reason":"missing_config"}' | tee -a "${LOG_FILE}"
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
  python -m scripts.validate_reward_plugin_boundary \
    --plugin rl/reward_builder_v2.py \
    2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m scripts.bp2_pipeline launch \
    --config "${CONFIG_FILE}" --stage grpo --dry-run --require-authorized \
    2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ] && [ "${SKIP_TRAINING}" != "1" ]; then
  if [ -n "${RESUME_FROM}" ]; then
    python -m scripts.bp2_pipeline launch \
      --config "${CONFIG_FILE}" --stage grpo --resume-from "${RESUME_FROM}" \
      2>&1 | tee -a "${LOG_FILE}" || status=$?
  else
    python -m scripts.bp2_pipeline launch \
      --config "${CONFIG_FILE}" --stage grpo \
      2>&1 | tee -a "${LOG_FILE}" || status=$?
  fi
fi

if [ "${status}" -eq 0 ] && [ "${SKIP_TRAINING}" = "1" ]; then
  echo '{"event":"bp4_e4_grpo_training_reuse","status":"PASS","reason":"training_already_completed_no_retraining"}' | tee -a "${LOG_FILE}"
fi

if [ "${status}" -eq 0 ]; then
  python -m scripts.bp2_pipeline inspect-artifacts \
    --output-dir "${OUTPUT_DIR}" --report-file "${ARTIFACT_FILE}" \
    2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -c 'import json; from pathlib import Path; p=Path("reports/bp4_e4_grpo_from_dpo/grpo_training_artifacts.json"); x=json.loads(p.read_text(encoding="utf-8")); assert x["checkpoint_count"]>0; assert x["latest_checkpoint"]["global_step"]>0; print(json.dumps({"event":"bp4_e4_grpo_artifact_check","status":"PASS","checkpoint_count":x["checkpoint_count"],"latest_checkpoint":x["latest_checkpoint"]["checkpoint"]},sort_keys=True))' 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ] && ! grep -q 'REWARD_V2_METRICS' "${LOG_FILE}"; then
  echo '{"event":"bp4_e4_grpo_reward_observability","status":"FAIL","reason":"missing_reward_v2_metrics"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  echo '{"event":"bp4_e4_grpo_runner_complete","status":"PASS","decision":"E4_GRPO_TRAINING_COMPLETE_EVALUATION_PENDING","training_started":true,"training_completed":true,"human_gold_used":false,"deployment_authorized":false}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"bp4_e4_grpo_runner_complete\",\"status\":\"FAIL\",\"exit_code\":${status},\"deployment_authorized\":false}" | tee -a "${LOG_FILE}"
fi

exit "${status}"
