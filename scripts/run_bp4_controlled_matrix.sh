#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${1:-configs/experiments/bp4_controlled_model_matrix_v1.yaml}"
LOG_FILE="${2:-new_plan/logs/bp4_controlled_model_matrix.log}"
status=0

mkdir -p "$(dirname "${LOG_FILE}")"
echo "{\"event\":\"bp4_runner_start\",\"work_package\":\"BP4_CONTROLLED_MODEL_MATRIX\",\"config\":\"${CONFIG_FILE}\",\"execution_policy\":\"matrix_engineering_and_authorization_design_only_no_training\"}" | tee "${LOG_FILE}"
python --version 2>&1 | tee -a "${LOG_FILE}"

if [ ! -f "${CONFIG_FILE}" ]; then
  echo '{"event":"bp4_input_check","status":"FAIL","reason":"missing_config"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  python -m unittest discover -v tests 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m scripts.bp4_matrix \
    --config "${CONFIG_FILE}" prepare 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ] && ! grep -q '"event": "bp4_gate_complete".*"status": "PASS"' "${LOG_FILE}"; then
  echo '{"event":"bp4_log_check","status":"FAIL","reason":"missing_gate_pass"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  python -c 'import json; from pathlib import Path; report=Path("reports/bp4_controlled_matrix"); gate=json.loads((report/"bp4_gate.json").read_text(encoding="utf-8")); plan=json.loads((report/"matrix_plan.json").read_text(encoding="utf-8")); request=json.loads((report/"authorization_candidates/bp2_sft.json").read_text(encoding="utf-8")); assert gate["status"] == "PASS"; assert gate["decision"] == "ACCEPT_BP4_MATRIX_ENGINEERING"; assert gate["training_started"] is False; assert plan["rows"][0]["id"] == "B1_V2" and plan["rows"][0]["status"] == "READY"; assert all(row["status"] == "BLOCKED" for row in plan["rows"][1:]); assert request["activation_status"] == "BLOCKED_MISSING_BASE_V2_EVALUATION"; assert "decision" not in request; print(json.dumps({"event":"bp4_independent_gate_check","status":"PASS","base_v2":"READY","training_stages":"BLOCKED","training_started":False},sort_keys=True))' 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  echo '{"event":"bp4_runner_complete","status":"PASS","decision":"ACCEPT_BP4_MATRIX_ENGINEERING","next_execution":"BP4_BASE_V2_EVALUATION","training_started":false}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"bp4_runner_complete\",\"status\":\"FAIL\",\"exit_code\":${status},\"training_started\":false}" | tee -a "${LOG_FILE}"
fi

exit "${status}"
