#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${1:-configs/experiments/bp3_reward_preferences_v1.yaml}"
LOG_FILE="${2:-new_plan/logs/bp3_reward_preferences.log}"
status=0

mkdir -p "$(dirname "${LOG_FILE}")"
echo "{\"event\":\"bp3_runner_start\",\"work_package\":\"BP3_REWARD_V2_AND_PREFERENCE_DATA\",\"config\":\"${CONFIG_FILE}\",\"execution_policy\":\"engineering_and_offline_validation_only_no_training\"}" | tee "${LOG_FILE}"
python --version 2>&1 | tee -a "${LOG_FILE}"

if [ ! -f "${CONFIG_FILE}" ]; then
  echo '{"event":"bp3_input_check","status":"FAIL","reason":"missing_config"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  python -m unittest discover -v tests 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m scripts.build_bp3_reward_preferences \
    --config "${CONFIG_FILE}" 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ] && ! grep -q '"event": "bp3_gate_complete".*"status": "PASS"' "${LOG_FILE}"; then
  echo '{"event":"bp3_log_check","status":"FAIL","reason":"missing_gate_pass"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  python -c 'import json; from pathlib import Path; report=Path("reports/bp3_keyword_v2"); names=("bp3_gate.json","reward_v2_gate.json","preference_gate.json"); gates=[json.loads((report/name).read_text(encoding="utf-8")) for name in names]; assert all(gate["status"] == "PASS" for gate in gates); assert json.loads((report/"bp3_gate.json").read_text(encoding="utf-8"))["training_started"] is False; print(json.dumps({"event":"bp3_independent_gate_check","status":"PASS","gates":list(names),"training_started":False},sort_keys=True))' 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  echo '{"event":"bp3_runner_complete","status":"PASS","decision":"ACCEPT_BP3_REWARD_AND_PREFERENCES","next_work_package":"BP4_CONTROLLED_MODEL_MATRIX","training_started":false}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"bp3_runner_complete\",\"status\":\"FAIL\",\"exit_code\":${status},\"training_started\":false}" | tee -a "${LOG_FILE}"
fi

exit "${status}"
