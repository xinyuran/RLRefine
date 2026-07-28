#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?served Base model id required}"
BASE_URL="${2:-http://127.0.0.1:8003}"
LOG_FILE="${3:-reports/second_schema/second_schema_smoke_test.log}"

mkdir -p "$(dirname "${LOG_FILE}")"
python -m unittest -v \
  tests.test_intent_routing_schema \
  tests.test_second_schema_assets \
  tests.test_second_schema_benchmark 2>&1 | tee "${LOG_FILE}"
python -m evaluation.second_schema_benchmark \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" 2>&1 | tee -a "${LOG_FILE}"
