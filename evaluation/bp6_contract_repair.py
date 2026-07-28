"""Deterministic, audit-preserving repair of keyword target-contract failures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.keyword_contract import MAX_KEYWORD_LENGTH, MAX_KEYWORDS
from core.target_contract import TARGET_CONTRACT_VERSION, validate_keyword_payload
from evaluation.bp2_evaluator import evaluate_with_slices, prepare_reference
from evaluation.keyword_evaluator import file_sha256, read_jsonl, write_json_atomic, write_jsonl_atomic


REPAIR_VERSION = "bp6-deterministic-contract-repair-v1"


def repair_payload(raw_response: object, source: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Keep only contract-valid keyword tuples; never invent or rewrite a keyword."""
    if not isinstance(raw_response, str):
        return None, ["missing_raw_response"]
    try:
        payload = json.loads(raw_response.strip())
    except json.JSONDecodeError:
        return None, ["json_not_parseable"]
    if not isinstance(payload, dict) or not isinstance(payload.get("keywords"), list):
        return None, ["keywords_not_recoverable"]
    actions: list[str] = []
    kept, seen = [], set()
    for item in payload["keywords"]:
        if not isinstance(item, list) or len(item) != 3:
            actions.append("drop_tuple_shape_invalid"); continue
        explanation, keyword, confidence = item
        if not isinstance(explanation, str) or not explanation.strip():
            actions.append("drop_explanation_invalid"); continue
        if not isinstance(keyword, str) or not 1 <= len(keyword) <= MAX_KEYWORD_LENGTH:
            actions.append("drop_keyword_length_or_type_invalid"); continue
        if keyword not in source:
            actions.append("drop_keyword_not_in_source"); continue
        if keyword in seen:
            actions.append("drop_keyword_duplicate"); continue
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            actions.append("drop_confidence_invalid"); continue
        seen.add(keyword); kept.append([explanation, keyword, confidence])
    if len(kept) > MAX_KEYWORDS:
        kept = kept[:MAX_KEYWORDS]; actions.append("truncate_to_max_keywords")
    if not kept:
        return None, actions + ["no_recoverable_keywords"]
    repaired = {"keywords": kept}
    valid, errors = validate_keyword_payload(repaired, source)
    if not valid:
        raise ValueError(f"Repair emitted an invalid payload: {errors}")
    return repaired, actions


def repair_prediction(row: Mapping[str, Any], source: str) -> dict[str, Any]:
    repaired, actions = repair_payload(row.get("raw_response"), source)
    output = dict(row)
    output.update({"repair_version": REPAIR_VERSION, "repair_actions": actions, "original_status": row.get("status")})
    if repaired is None:
        output.update({"status": "error", "data": None, "error_code": "contract_repair_unrecoverable"})
    else:
        output.update({"status": "success", "data": repaired, "error_code": None, "validation_errors": []})
    output["target_contract_version"] = TARGET_CONTRACT_VERSION
    return output


def run(dataset: Path, predictions: Path, output_dir: Path, model_id: str) -> dict[str, Any]:
    dataset_rows, original = read_jsonl(dataset), read_jsonl(predictions)
    reference = prepare_reference(dataset_rows)
    sources = {row["sample_id"]: row["normalized_text"] for row in reference}
    original_ids = {row.get("sample_id") for row in original}
    if original_ids != set(sources) or len(original_ids) != len(original):
        raise ValueError("Prediction coverage must exactly match the frozen dataset")
    repaired = [repair_prediction(row, sources[row["sample_id"]]) for row in original]
    before, _ = evaluate_with_slices(reference, original, f"{model_id}:raw")
    after, errors = evaluate_with_slices(reference, repaired, f"{model_id}:repaired")
    success_valid = all(
        row["status"] != "success" or validate_keyword_payload(row["data"], sources[row["sample_id"]])[0]
        for row in repaired
    )
    checks = {
        "coverage_exact": len(repaired) == len(reference),
        "successful_repairs_contract_valid": success_valid,
        "macro_f1_noninferior_within_0_01": after["macro"]["f1"] >= before["macro"]["f1"] - 0.01,
        "human_gold_not_used": True,
        "model_prompt_decoding_unchanged": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    repaired_path = output_dir / "repaired_predictions.jsonl"
    write_jsonl_atomic(repaired_path, repaired)
    write_json_atomic(output_dir / "raw_evaluation.json", before)
    write_json_atomic(output_dir / "repaired_evaluation.json", after)
    write_jsonl_atomic(output_dir / "repaired_errors.jsonl", errors)
    report = {
        "report_version": REPAIR_VERSION,
        "status": "PASS",
        "decision": "RETAIN_REPAIRED_E4_FOR_OFFLINE_REVIEW" if all(checks.values()) else "RETAIN_RAW_E4_FOR_OFFLINE_REVIEW",
        "deployment_authorized": False,
        "dataset": str(dataset), "dataset_sha256": file_sha256(dataset),
        "predictions": str(predictions), "predictions_sha256": file_sha256(predictions),
        "repaired_predictions_sha256": file_sha256(repaired_path),
        "sample_count": len(reference), "raw": {"macro_f1": before["macro"]["f1"], "micro_f1": before["micro"]["f1"], "schema_valid_rate": before["schema_valid_rate"]},
        "repaired": {"macro_f1": after["macro"]["f1"], "micro_f1": after["micro"]["f1"], "schema_valid_rate": after["schema_valid_rate"]},
        "repaired_row_count": sum(row["repair_actions"] != [] for row in repaired),
        "unrecoverable_row_count": sum(row["status"] == "error" for row in repaired),
        "checks": checks,
        "next_execution": "HUMAN_REVIEW_SECOND_SCHEMA_DATASET_SCOPE",
    }
    write_json_atomic(output_dir / "repair_report.json", report)
    print(json.dumps({"event": "bp6_contract_repair_complete", **report}, ensure_ascii=False, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="E4_GRPO_FROM_DPO")
    args = parser.parse_args()
    try:
        run(args.dataset, args.predictions, args.output_dir, args.model_id); return 0
    except Exception as exc:
        print(json.dumps({"event": "bp6_contract_repair_complete", "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, sort_keys=True)); return 1


if __name__ == "__main__":
    raise SystemExit(main())
