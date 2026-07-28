#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="${1:-new_plan/logs/bp6_contract_repair.log}"
mkdir -p "$(dirname "${LOG_FILE}")"
python -m unittest discover -v tests 2>&1 | tee "${LOG_FILE}"
python -m evaluation.bp6_contract_repair --dataset data/canonical/keyword_v2/dev.jsonl --predictions reports/bp4_e4_grpo_from_dpo/evaluation/dev_predictions.jsonl --output-dir reports/bp6_contract_repair/e4_dev 2>&1 | tee -a "${LOG_FILE}"
python -m evaluation.bp6_contract_repair --dataset data/canonical/keyword_v2/challenge.jsonl --predictions reports/bp4_e4_grpo_from_dpo/evaluation/challenge_predictions.jsonl --output-dir reports/bp6_contract_repair/e4_challenge 2>&1 | tee -a "${LOG_FILE}"
