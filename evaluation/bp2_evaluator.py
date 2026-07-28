"""Unified dev/challenge evaluator adapter for every BP2 training stage."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from evaluation.keyword_evaluator import (
    evaluate_records,
    file_sha256,
    read_jsonl,
    render_markdown,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from scripts.build_bp1_assets import source_and_payload


def prepare_reference(rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    prepared = []
    seen = set()
    for row_number, row in enumerate(rows, 1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"Invalid/duplicate sample_id at row {row_number}")
        seen.add(sample_id)
        source, payload = source_and_payload(row)
        prepared.append(
            {
                "sample_id": sample_id,
                "normalized_text": source,
                "gold": {"keywords": [item[1] for item in payload["keywords"]]},
                "label_origin": row.get("label_origin"),
                "split": row.get("split"),
                "challenge_slices": row.get("challenge_slices") or [],
            }
        )
    return sorted(prepared, key=lambda row: row["sample_id"])


def _resource_metrics(predictions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def mean(field: str) -> float | None:
        values = [
            float(row[field])
            for row in predictions
            if isinstance(row.get(field), (int, float))
        ]
        return sum(values) / len(values) if values else None

    return {
        "mean_input_tokens": mean("input_tokens"),
        "mean_output_tokens": mean("output_tokens"),
        "mean_request_seconds": mean("request_seconds"),
    }


def evaluate_with_slices(
    reference: Sequence[Dict[str, Any]],
    predictions: Sequence[Dict[str, Any]],
    model_id: str,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    overall, errors = evaluate_records(reference, predictions, model_id)
    prediction_by_id = {row["sample_id"]: row for row in predictions}
    grouped = defaultdict(list)
    for row in reference:
        for slice_name in row.get("challenge_slices") or []:
            grouped[slice_name].append(row)
    slice_reports = {}
    for slice_name, slice_reference in sorted(grouped.items()):
        slice_predictions = [
            prediction_by_id[row["sample_id"]] for row in slice_reference
        ]
        report, _ = evaluate_records(
            slice_reference, slice_predictions, f"{model_id}:{slice_name}"
        )
        slice_reports[slice_name] = {
            "sample_count": report["sample_count"],
            "micro_f1": report["micro"]["f1"],
            "macro_f1": report["macro"]["f1"],
            "schema_valid_rate": report["schema_valid_rate"],
            "hallucination_keyword_rate": report["hallucination"]["keyword_rate"],
            "error_sample_count": report["error_sample_count"],
        }
    overall["resource_metrics"] = _resource_metrics(predictions)
    overall["challenge_slices"] = slice_reports
    overall["selection_policy"] = (
        "teacher_dev_for_selection_challenge_for_diagnostics_human_gold_forbidden"
    )
    return overall, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-reference")
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--reference", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--model-id", required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare-reference":
            rows = prepare_reference(read_jsonl(args.dataset))
            write_jsonl_atomic(args.output, rows)
            print(
                json.dumps(
                    {
                        "event": "bp2_reference_prepare",
                        "status": "PASS",
                        "rows": len(rows),
                        "output": str(args.output),
                        "sha256": file_sha256(args.output),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        report, errors = evaluate_with_slices(
            read_jsonl(args.reference),
            read_jsonl(args.predictions),
            args.model_id,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_file = args.output_dir / "evaluation.json"
        error_file = args.output_dir / "errors.jsonl"
        markdown_file = args.output_dir / "evaluation.md"
        write_json_atomic(report_file, report)
        write_jsonl_atomic(error_file, errors)
        write_text_atomic(markdown_file, render_markdown(report))
        print(
            json.dumps(
                {
                    "event": "bp2_evaluation_complete",
                    "status": "PASS",
                    "report_file": str(report_file),
                    "report_sha256": file_sha256(report_file),
                    "errors_file": str(error_file),
                    "errors_sha256": file_sha256(error_file),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "event": "bp2_evaluator",
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
