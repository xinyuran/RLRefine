#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from evaluation.keyword_evaluator import (
    evaluate_records,
    file_sha256,
    read_jsonl,
    render_markdown,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)


EXPECTED_GOLD_SHA256 = "0df4d88e8bc09d62f867ef8a4b43fe631ed9e357b79b55e022948f9b75e06bf6"


def emit(event, **payload):
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold-file",
        type=Path,
        default=Path("data/canonical/keyword_v1/gold/v1/gold_test.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/evaluation_contract/v1"))
    args = parser.parse_args()

    gold_hash = file_sha256(args.gold_file)
    if gold_hash != EXPECTED_GOLD_SHA256:
        emit(
            "p1_evaluation_contract_complete",
            status="FAIL",
            reason="gold_sha256_mismatch",
            expected=EXPECTED_GOLD_SHA256,
            actual=gold_hash,
        )
        return 1

    gold_rows = read_jsonl(args.gold_file)
    predictions = [
        {
            "sample_id": row["sample_id"],
            "status": "success",
            "data": {
                "keywords": [["gold-self-check", keyword, 1.0] for keyword in row["gold"]["keywords"]]
            },
        }
        for row in gold_rows
    ]
    report, errors = evaluate_records(gold_rows, predictions, "reference-self-check-not-model-result")
    report["label"] = "reference_self_check_not_model_result"
    report["gold_sha256"] = gold_hash
    report["prediction_source"] = "derived_from_gold_for_metric_contract_validation_only"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_file = args.output_dir / "contract_self_predictions.jsonl"
    report_file = args.output_dir / "evaluation_report.json"
    markdown_file = args.output_dir / "evaluation_report.md"
    error_file = args.output_dir / "evaluation_errors.jsonl"
    write_jsonl_atomic(prediction_file, predictions)
    write_json_atomic(report_file, report)
    write_text_atomic(markdown_file, render_markdown(report))
    write_jsonl_atomic(error_file, errors)

    checks = {
        "gold_rows_496": len(gold_rows) == 496,
        "schema_valid_rate_one": report["schema_valid_rate"] == 1.0,
        "exact_set_match_rate_one": report["exact_set_match_rate"] == 1.0,
        "micro_f1_one": report["micro"]["f1"] == 1.0,
        "macro_f1_one": report["macro"]["f1"] == 1.0,
        "hallucination_rate_zero": report["hallucination"]["keyword_rate"] == 0.0,
        "error_rows_zero": len(errors) == 0,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    emit(
        "p1_evaluation_contract_complete",
        status=status,
        label=report["label"],
        checks=checks,
        gold_file=str(args.gold_file),
        gold_sha256=gold_hash,
        prediction_file=str(prediction_file),
        prediction_sha256=file_sha256(prediction_file),
        report_file=str(report_file),
        report_sha256=file_sha256(report_file),
        markdown_file=str(markdown_file),
        markdown_sha256=file_sha256(markdown_file),
        errors_file=str(error_file),
        errors_sha256=file_sha256(error_file),
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
