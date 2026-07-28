#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:?served E4 model id required}"
BASE_URL="${2:-http://127.0.0.1:8002}"
LOG_FILE="${3:-new_plan/logs/bp5_e4_serving_benchmark.log}"
mkdir -p "$(dirname "${LOG_FILE}")"
python -m unittest discover -v tests 2>&1 | tee "${LOG_FILE}"
python -m evaluation.bp5_serving_benchmark --model "${MODEL}" --base-url "${BASE_URL}" 2>&1 | tee -a "${LOG_FILE}"
