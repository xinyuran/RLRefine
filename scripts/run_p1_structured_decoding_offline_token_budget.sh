#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-Qwen/Qwen2.5-7B-Instruct}"
DEV_FILE="${2:-data/canonical/keyword_v1/splits/grpo_dev.jsonl}"
PRIOR_DIR="${3:-reports/baselines/qwen2_5_7b_dev/v5_structured_decoding}"
EXPERIMENT_DIR="${4:-reports/baselines/qwen2_5_7b_dev/v7_offline_deterministic_768}"
LOG_FILE="${5:-new_plan/logs/p1_structured_decoding_offline_token_budget.log}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
status=0

mkdir -p "$(dirname "${LOG_FILE}")" "${EXPERIMENT_DIR}"

echo "{\"event\":\"p1_structured_decoding_offline_token_budget_runner_start\",\"label\":\"teacher_dev_offline_deterministic_token_budget_not_human_gold_test\",\"model_path\":\"${MODEL_PATH}\",\"dev_file\":\"${DEV_FILE}\",\"prior_dir\":\"${PRIOR_DIR}\",\"experiment_dir\":\"${EXPERIMENT_DIR}\",\"cuda_visible_devices\":\"${CUDA_VISIBLE_DEVICES}\",\"vllm_enable_v1_multiprocessing\":\"${VLLM_ENABLE_V1_MULTIPROCESSING}\",\"single_variable\":\"max_tokens_512_to_768\"}" | tee "${LOG_FILE}"
python --version 2>&1 | tee -a "${LOG_FILE}"

for required_path in \
  "${MODEL_PATH}" \
  "${DEV_FILE}" \
  "${PRIOR_DIR}/dev_teacher_reference.jsonl" \
  "${PRIOR_DIR}/b1_vllm_unconstrained_v1_predictions.jsonl" \
  "${PRIOR_DIR}/b1_vllm_json_schema_v1_predictions.jsonl"; do
  if [ ! -e "${required_path}" ]; then
    echo "{\"event\":\"p1_structured_decoding_offline_token_budget_input\",\"status\":\"FAIL\",\"reason\":\"missing_path\",\"path\":\"${required_path}\"}" | tee -a "${LOG_FILE}"
    status=1
  fi
done

if [ "${status}" -eq 0 ]; then
  python -m unittest -v \
    tests.test_keyword_schema_contract \
    tests.test_keyword_evaluator \
    tests.test_baseline_inference \
    tests.test_baseline_diagnostics \
    tests.test_structured_decoding \
    tests.test_structured_decoding_token_budget \
    tests.test_structured_decoding_feedback 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -c 'import json, os, torch, vllm; print(json.dumps({"event":"p1_structured_decoding_offline_token_budget_environment","torch":torch.__version__,"vllm":vllm.__version__,"cuda_available":torch.cuda.is_available(),"cuda_device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"vllm_enable_v1_multiprocessing":os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")}, ensure_ascii=False))' 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m evaluation.structured_decoding_offline_token_budget_inference \
    --model-path "${MODEL_PATH}" \
    --dev-file "${DEV_FILE}" \
    --prior-dir "${PRIOR_DIR}" \
    --output-dir "${EXPERIMENT_DIR}" 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m evaluation.structured_decoding_offline_token_budget_diagnostics \
    --prior-dir "${PRIOR_DIR}" \
    --experiment-dir "${EXPERIMENT_DIR}" \
    --bootstrap-iterations 10000 \
    --seed 42 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m scripts.package_structured_decoding_feedback \
    --experiment-dir "${EXPERIMENT_DIR}" 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ] && ! grep -q '"event": "p1_structured_decoding_offline_token_budget_gate_complete".*"status": "PASS"' "${LOG_FILE}"; then
  echo '{"event":"p1_structured_decoding_offline_token_budget_log_check","status":"FAIL","reason":"missing_gate_pass"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ] && ! grep -q '"event": "p1_structured_decoding_feedback_packet_complete".*"status": "PASS"' "${LOG_FILE}"; then
  echo '{"event":"p1_structured_decoding_offline_token_budget_log_check","status":"FAIL","reason":"missing_feedback_packet_pass"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  echo '{"event":"p1_structured_decoding_offline_token_budget_runner_complete","status":"PASS"}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"p1_structured_decoding_offline_token_budget_runner_complete\",\"status\":\"FAIL\",\"exit_code\":${status}}" | tee -a "${LOG_FILE}"
fi

exit "${status}"
