#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-Qwen/Qwen2.5-7B-Instruct}"
DEV_FILE="${2:-data/canonical/keyword_v1/splits/grpo_dev.jsonl}"
BASELINE_DIR="${3:-reports/baselines/qwen2_5_7b_dev/v1}"
CANDIDATE_DIR="${4:-reports/baselines/qwen2_5_7b_dev/v3}"
LOG_FILE="${5:-new_plan/logs/p1_prompt_optimization_b3.log}"

export CANDIDATE_VARIANT="b3_balanced_v1"
bash scripts/run_p1_prompt_optimization.sh \
  "${MODEL_PATH}" "${DEV_FILE}" "${BASELINE_DIR}" "${CANDIDATE_DIR}" "${LOG_FILE}"
