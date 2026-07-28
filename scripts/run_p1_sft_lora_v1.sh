#!/usr/bin/env bash
set -euo pipefail

MANIFEST_FILE="${1:-reports/lineage/p1_sft_lora_v1/run_manifest.json}"
LOG_FILE="${2:-new_plan/logs/p1_sft_lora_v1.log}"
SWIFT_CONFIG="reports/lineage/p1_sft_lora_v1/swift_sft_config.yaml"
TRAIN_OUTPUT="reports/training/p1_sft_lora_v1"
mkdir -p "$(dirname "${LOG_FILE}")"
status=0

echo "{\"event\":\"p1_sft_runner_start\",\"manifest_file\":\"${MANIFEST_FILE}\",\"swift_config\":\"${SWIFT_CONFIG}\",\"server_repo\":\".\"}" | tee "${LOG_FILE}"
python --version 2>&1 | tee -a "${LOG_FILE}"

if [ ! -f "${MANIFEST_FILE}" ] || [ ! -f "${SWIFT_CONFIG}" ]; then
  echo '{"event":"p1_sft_runner_input","status":"FAIL","reason":"missing_manifest_or_swift_config"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  python -m scripts.validate_sft_launch --manifest "${MANIFEST_FILE}" 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  echo '{"event":"p1_sft_training_start","command":["python","-m","scripts.swift_yaml_launcher","--config","reports/lineage/p1_sft_lora_v1/swift_sft_config.yaml"]}' | tee -a "${LOG_FILE}"
  python -m scripts.swift_yaml_launcher \
    --config "${SWIFT_CONFIG}" 2>&1 | tee -a "${LOG_FILE}" || status=${PIPESTATUS[0]}
fi

if [ "${status}" -eq 0 ]; then
  checkpoint_count="$(find "${TRAIN_OUTPUT}" -type d -name 'checkpoint-*' | wc -l | tr -d ' ')"
  trainer_state_count="$(find "${TRAIN_OUTPUT}" -type f -name 'trainer_state.json' | wc -l | tr -d ' ')"
  if [ "${checkpoint_count}" -lt 1 ] || [ "${trainer_state_count}" -lt 1 ]; then
    echo "{\"event\":\"p1_sft_artifact_check\",\"status\":\"FAIL\",\"checkpoint_count\":${checkpoint_count},\"trainer_state_count\":${trainer_state_count}}" | tee -a "${LOG_FILE}"
    status=1
  else
    echo "{\"event\":\"p1_sft_artifact_check\",\"status\":\"PASS\",\"checkpoint_count\":${checkpoint_count},\"trainer_state_count\":${trainer_state_count}}" | tee -a "${LOG_FILE}"
  fi
fi

if [ "${status}" -eq 0 ]; then
  echo '{"event":"p1_sft_runner_complete","status":"PASS","next":"inspect_training_log_and_select_best_eval_loss_checkpoint"}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"p1_sft_runner_complete\",\"status\":\"FAIL\",\"exit_code\":${status}}" | tee -a "${LOG_FILE}"
fi

exit "${status}"
