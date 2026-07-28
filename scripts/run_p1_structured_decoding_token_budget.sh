#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-Qwen/Qwen2.5-7B-Instruct}"
DEV_FILE="${2:-data/canonical/keyword_v1/splits/grpo_dev.jsonl}"
PRIOR_DIR="${3:-reports/baselines/qwen2_5_7b_dev/v5_structured_decoding}"
EXPERIMENT_DIR="${4:-reports/baselines/qwen2_5_7b_dev/v6_structured_decoding_768}"
LOG_FILE="${5:-new_plan/logs/p1_structured_decoding_token_budget.log}"
PORT="${PORT:-8002}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-7b-instruct}"
WORKERS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BASE_URL="http://127.0.0.1:${PORT}"
SERVER_LOG="${EXPERIMENT_DIR}/vllm_server.log"
SERVER_PID=""
status=0

mkdir -p "$(dirname "${LOG_FILE}")" "${EXPERIMENT_DIR}"

cleanup() {
  if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "{\"event\":\"p1_structured_decoding_token_budget_runner_start\",\"label\":\"teacher_dev_structured_decoding_token_budget_not_human_gold_test\",\"model_path\":\"${MODEL_PATH}\",\"dev_file\":\"${DEV_FILE}\",\"prior_dir\":\"${PRIOR_DIR}\",\"experiment_dir\":\"${EXPERIMENT_DIR}\",\"cuda_visible_devices\":\"${CUDA_VISIBLE_DEVICES}\",\"port\":${PORT},\"workers\":${WORKERS},\"single_variable\":\"max_tokens_512_to_768\"}" | tee "${LOG_FILE}"
python --version 2>&1 | tee -a "${LOG_FILE}"

for required_path in \
  "${MODEL_PATH}" \
  "${DEV_FILE}" \
  "${PRIOR_DIR}/dev_teacher_reference.jsonl" \
  "${PRIOR_DIR}/b1_vllm_unconstrained_v1_predictions.jsonl" \
  "${PRIOR_DIR}/b1_vllm_json_schema_v1_predictions.jsonl"; do
  if [ ! -e "${required_path}" ]; then
    echo "{\"event\":\"p1_structured_decoding_token_budget_input\",\"status\":\"FAIL\",\"reason\":\"missing_path\",\"path\":\"${required_path}\"}" | tee -a "${LOG_FILE}"
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
    tests.test_structured_decoding_token_budget 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -c 'import json, openai, vllm; print(json.dumps({"event":"p1_structured_decoding_token_budget_environment","openai":openai.__version__,"vllm":vllm.__version__}, ensure_ascii=False))' 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  echo '{"event":"p1_structured_decoding_token_budget_server_start"}' | tee -a "${LOG_FILE}"
  vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --dtype bfloat16 \
    --seed 42 \
    --max-model-len 4864 \
    --max-num-seqs "${WORKERS}" \
    --gpu-memory-utilization 0.85 \
    --disable-log-requests >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!

  ready=0
  for _ in $(seq 1 180); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      break
    fi
    if curl --silent --fail "${BASE_URL}/v1/models" >/dev/null; then
      ready=1
      break
    fi
    sleep 5
  done
  if [ "${ready}" -ne 1 ]; then
    echo "{\"event\":\"p1_structured_decoding_token_budget_server_ready\",\"status\":\"FAIL\",\"server_log\":\"${SERVER_LOG}\"}" | tee -a "${LOG_FILE}"
    tail -n 80 "${SERVER_LOG}" 2>/dev/null | tee -a "${LOG_FILE}" || true
    status=1
  else
    echo "{\"event\":\"p1_structured_decoding_token_budget_server_ready\",\"status\":\"PASS\",\"server_pid\":${SERVER_PID}}" | tee -a "${LOG_FILE}"
  fi
fi

if [ "${status}" -eq 0 ]; then
  python -m evaluation.structured_decoding_token_budget_inference \
    --base-url "${BASE_URL}" \
    --model "${SERVED_MODEL_NAME}" \
    --dev-file "${DEV_FILE}" \
    --prior-dir "${PRIOR_DIR}" \
    --output-dir "${EXPERIMENT_DIR}" \
    --workers "${WORKERS}" 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m evaluation.structured_decoding_token_budget_diagnostics \
    --prior-dir "${PRIOR_DIR}" \
    --experiment-dir "${EXPERIMENT_DIR}" \
    --bootstrap-iterations 10000 \
    --seed 42 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ] && ! grep -q '"event": "p1_structured_decoding_token_budget_gate_complete".*"status": "PASS"' "${LOG_FILE}"; then
  echo '{"event":"p1_structured_decoding_token_budget_log_check","status":"FAIL","reason":"missing_gate_pass"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  echo '{"event":"p1_structured_decoding_token_budget_runner_complete","status":"PASS"}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"p1_structured_decoding_token_budget_runner_complete\",\"status\":\"FAIL\",\"exit_code\":${status}}" | tee -a "${LOG_FILE}"
fi

exit "${status}"
