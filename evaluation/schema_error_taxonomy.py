"""Deterministic, mutually-exclusive taxonomy for BP4 schema failures."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.keyword_evaluator import file_sha256, read_jsonl, write_json_atomic, write_jsonl_atomic


TAXONOMY_VERSION = "bp5-schema-error-taxonomy-v1"


def classify_prediction(row: Mapping[str, Any]) -> str | None:
    """Return the first violated contract layer, or None for a valid output."""
    if row.get("status") == "success":
        return None
    raw = row.get("raw_response")
    if not isinstance(raw, str):
        return "missing_raw_response"
    try:
        payload = json.loads(raw.strip())
    except json.JSONDecodeError:
        return "json_not_parseable"
    if not isinstance(payload, dict):
        return "top_level_not_object"
    if set(payload) != {"keywords"}:
        return "top_level_keys_invalid"
    keywords = payload.get("keywords")
    if not isinstance(keywords, list):
        return "keywords_not_array"
    if not 1 <= len(keywords) <= 15:
        return "keyword_count_out_of_range"
    for item in keywords:
        if not isinstance(item, list) or len(item) != 3:
            return "tuple_shape_invalid"
        category, keyword, confidence = item
        if not isinstance(category, str) or not category:
            return "category_invalid"
        if not isinstance(keyword, str) or not 1 <= len(keyword) <= 4:
            return "keyword_length_or_type_invalid"
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return "confidence_not_numeric"
        if not 0 <= confidence <= 1:
            return "confidence_out_of_range"
    # The BP4 validator may additionally reject source-faithfulness or duplicates.
    errors = " ".join(str(value) for value in row.get("validation_errors") or [])
    if "source" in errors.lower() or "原文" in errors:
        return "keyword_not_in_source"
    if "duplicate" in errors.lower() or "重复" in errors:
        return "duplicate_keyword"
    return str(row.get("error_code") or "unclassified_contract_failure")


def analyze(predictions: Sequence[Mapping[str, Any]], expected_errors: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    invalid = []
    seen = set()
    for row in predictions:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError("Predictions must have unique non-empty sample_id values")
        seen.add(sample_id)
        category = classify_prediction(row)
        if category is not None:
            invalid.append({
                "sample_id": sample_id,
                "category": category,
                "error_code": row.get("error_code"),
                "validation_errors": row.get("validation_errors") or [],
                "finish_reason": row.get("finish_reason"),
                "output_tokens": row.get("output_tokens"),
            })
    if expected_errors is not None and len(invalid) != expected_errors:
        raise ValueError(f"Expected {expected_errors} schema-invalid rows, found {len(invalid)}")
    counts = Counter(item["category"] for item in invalid)
    report = {
        "report_version": TAXONOMY_VERSION,
        "sample_count": len(predictions),
        "schema_error_count": len(invalid),
        "schema_error_rate": len(invalid) / len(predictions) if predictions else 0.0,
        "category_counts": dict(sorted(counts.items())),
        "mutually_exclusive": True,
        "classification_policy": "first_violated_contract_layer",
    }
    return report, invalid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-errors", type=int)
    args = parser.parse_args()
    try:
        report, rows = analyze(read_jsonl(args.predictions), args.expected_errors)
        report.update({"predictions": str(args.predictions), "predictions_sha256": file_sha256(args.predictions)})
        write_json_atomic(args.output_dir / "schema_error_taxonomy.json", report)
        write_jsonl_atomic(args.output_dir / "schema_error_taxonomy_rows.jsonl", rows)
        print(json.dumps({"event": "bp5_schema_error_taxonomy_complete", "status": "PASS", **report}, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"event": "bp5_schema_error_taxonomy_complete", "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
