"""Paired diagnostics for the B0/B1 development-set baseline run."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from evaluation.keyword_evaluator import (
    evaluate_records,
    file_sha256,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from examples.keyword_extraction.schema import create_keyword_schema


EXPECTED_HASHES = {
    "reference": "2c4cd0e5eaf14dc4ade0ac857e70d5669d9ea4746608c46c6bf3df6df3d83dfe",
    "b0": "fde955224545254a8e8261dd55bb6a304e69026bd5fdce1e2b22b4c857a61dcb",
    "b1": "deec1c9138efa09910c6cd202f92d2c1bf3979b0a9f09db197bef7a5f4d59abc",
}


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True), flush=True)


def _index_unique(rows: Sequence[Dict[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
    indexed = {}
    for row_number, row in enumerate(rows, 1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in indexed:
            raise ValueError(f"Invalid or duplicate sample_id in {label} row {row_number}")
        indexed[sample_id] = row
    return indexed


def _sample_f1(predicted: set, gold: set) -> float:
    if not predicted or not gold:
        return 0.0
    precision = len(predicted & gold) / len(predicted)
    recall = len(predicted & gold) / len(gold)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _schema_issue(row: Dict[str, Any]) -> str:
    if row.get("status") != "error":
        return "none"
    error_code = row.get("error_code")
    if error_code == "json_parse_failed":
        return "json_parse_failed"
    errors = " ".join(str(item) for item in row.get("validation_errors") or [])
    if "numeric type" in errors:
        return "confidence_not_numeric"
    if "must not be greater" in errors or "must not be less" in errors:
        return "numeric_or_count_out_of_range"
    if "exactly" in errors or "tuple array type" in errors:
        return "tuple_shape_invalid"
    if "length" in errors:
        return "string_or_array_length_invalid"
    return str(error_code or "other_error")


def _prediction_view(row: Dict[str, Any], source_text: str) -> Dict[str, Any]:
    data = row.get("data")
    valid, _ = create_keyword_schema().validate(data if isinstance(data, dict) else {})
    schema_valid = row.get("status") in {"success", "fallback"} and valid
    keywords = [item[1] for item in data["keywords"]] if schema_valid else []
    input_tokens = row.get("input_tokens")
    output_tokens = row.get("output_tokens")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        raise ValueError(f"Missing token counts for sample_id={row.get('sample_id')}")
    return {
        "keywords": set(keywords),
        "schema_valid": schema_valid,
        "schema_issue": _schema_issue(row),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "hallucinated": {keyword for keyword in keywords if keyword not in source_text},
    }


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    index = round((len(sorted_values) - 1) * probability)
    return sorted_values[index]


def _bootstrap_mean_ci(
    values: Sequence[float], iterations: int, seed: int
) -> Tuple[float, float]:
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(iterations)
    )
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _top(counter: Counter, limit: int = 20) -> List[Dict[str, Any]]:
    return [{"keyword": keyword, "count": count} for keyword, count in counter.most_common(limit)]


def analyze(
    reference_rows: Sequence[Dict[str, Any]],
    b0_rows: Sequence[Dict[str, Any]],
    b1_rows: Sequence[Dict[str, Any]],
    max_new_tokens: int,
    bootstrap_iterations: int,
    seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    reference = _index_unique(reference_rows, "reference")
    b0 = _index_unique(b0_rows, "b0")
    b1 = _index_unique(b1_rows, "b1")
    if not reference:
        raise ValueError("Reference rows must not be empty")
    if set(reference) != set(b0) or set(reference) != set(b1):
        raise ValueError("B0/B1/reference sample coverage mismatch")

    evaluator_reports = {
        "b0_simple": evaluate_records(reference_rows, b0_rows, "b0_simple")[0],
        "b1_structured_v3": evaluate_records(reference_rows, b1_rows, "b1_structured_v3")[0],
    }
    paired_rows = []
    deltas = []
    win_counts = Counter()
    issue_counts = {"b0_simple": Counter(), "b1_structured_v3": Counter()}
    fp_counts = {"b0_simple": Counter(), "b1_structured_v3": Counter()}
    fn_counts = {"b0_simple": Counter(), "b1_structured_v3": Counter()}
    hallucination_counts = {"b0_simple": Counter(), "b1_structured_v3": Counter()}
    token_totals = {"b0_simple": [0, 0], "b1_structured_v3": [0, 0]}
    truncation_counts = Counter()

    for sample_id in sorted(reference):
        reference_row = reference[sample_id]
        source_text = reference_row.get("normalized_text") or ""
        gold = set(reference_row["gold"]["keywords"])
        views = {
            "b0_simple": _prediction_view(b0[sample_id], source_text),
            "b1_structured_v3": _prediction_view(b1[sample_id], source_text),
        }
        scores = {}
        for variant, view in views.items():
            scores[variant] = _sample_f1(view["keywords"], gold)
            issue_counts[variant][view["schema_issue"]] += 1
            fp_counts[variant].update(view["keywords"] - gold)
            fn_counts[variant].update(gold - view["keywords"])
            hallucination_counts[variant].update(view["hallucinated"])
            token_totals[variant][0] += view["input_tokens"]
            token_totals[variant][1] += view["output_tokens"]
            if view["output_tokens"] >= max_new_tokens:
                truncation_counts[variant] += 1
        delta = scores["b1_structured_v3"] - scores["b0_simple"]
        deltas.append(delta)
        outcome = "b1_win" if delta > 0 else "b0_win" if delta < 0 else "tie"
        win_counts[outcome] += 1
        paired_rows.append({
            "sample_id": sample_id,
            "b0_f1": scores["b0_simple"],
            "b1_f1": scores["b1_structured_v3"],
            "f1_delta_b1_minus_b0": delta,
            "outcome": outcome,
            "b0_schema_issue": views["b0_simple"]["schema_issue"],
            "b1_schema_issue": views["b1_structured_v3"]["schema_issue"],
            "b0_output_tokens": views["b0_simple"]["output_tokens"],
            "b1_output_tokens": views["b1_structured_v3"]["output_tokens"],
        })

    ci_low, ci_high = _bootstrap_mean_ci(deltas, bootstrap_iterations, seed)
    sample_count = len(reference)
    variant_diagnostics = {}
    for variant in ("b0_simple", "b1_structured_v3"):
        report = evaluator_reports[variant]
        variant_diagnostics[variant] = {
            "schema_valid_rate": report["schema_valid_rate"],
            "exact_set_match_rate": report["exact_set_match_rate"],
            "micro_f1": report["micro"]["f1"],
            "macro_f1": report["macro"]["f1"],
            "hallucination_keyword_rate": report["hallucination"]["keyword_rate"],
            "mean_input_tokens": token_totals[variant][0] / sample_count,
            "mean_output_tokens": token_totals[variant][1] / sample_count,
            "max_token_output_count": truncation_counts[variant],
            "schema_issue_counts": dict(sorted(issue_counts[variant].items())),
            "top_false_positive": _top(fp_counts[variant]),
            "top_false_negative": _top(fn_counts[variant]),
            "top_hallucinated": _top(hallucination_counts[variant]),
        }

    input_ratio = (
        variant_diagnostics["b1_structured_v3"]["mean_input_tokens"]
        / variant_diagnostics["b0_simple"]["mean_input_tokens"]
    )
    output_ratio = (
        variant_diagnostics["b1_structured_v3"]["mean_output_tokens"]
        / variant_diagnostics["b0_simple"]["mean_output_tokens"]
    )
    checks = {
        "b1_macro_f1_delta_positive": sum(deltas) / sample_count > 0,
        "b1_paired_bootstrap_ci_excludes_zero": ci_low > 0,
        "b1_micro_f1_higher": variant_diagnostics["b1_structured_v3"]["micro_f1"] > variant_diagnostics["b0_simple"]["micro_f1"],
        "b1_schema_valid_rate_higher": variant_diagnostics["b1_structured_v3"]["schema_valid_rate"] > variant_diagnostics["b0_simple"]["schema_valid_rate"],
        "b1_hallucination_rate_lower": variant_diagnostics["b1_structured_v3"]["hallucination_keyword_rate"] < variant_diagnostics["b0_simple"]["hallucination_keyword_rate"],
        "no_max_token_outputs": sum(truncation_counts.values()) == 0,
    }
    report = {
        "report_version": "paired-baseline-diagnostics-v1",
        "label": "teacher_dev_diagnostics_not_human_gold_test",
        "sample_count": sample_count,
        "bootstrap_iterations": bootstrap_iterations,
        "seed": seed,
        "paired": {
            "macro_f1_delta_b1_minus_b0": sum(deltas) / sample_count,
            "bootstrap_95_ci": [ci_low, ci_high],
            "outcomes": dict(sorted(win_counts.items())),
        },
        "token_cost_ratio_b1_over_b0": {
            "input": input_ratio,
            "output": output_ratio,
        },
        "variants": variant_diagnostics,
        "checks": checks,
        "provisional_decision": (
            "b1_quality_winner_optimize_prompt_cost_before_freeze"
            if all(value for key, value in checks.items() if key != "no_max_token_outputs")
            else "diagnosis_required_before_prompt_freeze"
        ),
    }
    return report, paired_rows


def render_markdown(report: Dict[str, Any]) -> str:
    paired = report["paired"]
    b0 = report["variants"]["b0_simple"]
    b1 = report["variants"]["b1_structured_v3"]
    return "\n".join([
        "# Paired B0/B1 Development Diagnostics",
        "",
        f"- Label: `{report['label']}`",
        f"- Samples: {report['sample_count']}",
        f"- Paired macro-F1 delta (B1-B0): {paired['macro_f1_delta_b1_minus_b0']:.6f}",
        f"- Bootstrap 95% CI: [{paired['bootstrap_95_ci'][0]:.6f}, {paired['bootstrap_95_ci'][1]:.6f}]",
        f"- Outcomes: {json.dumps(paired['outcomes'], ensure_ascii=False, sort_keys=True)}",
        f"- B0/B1 micro-F1: {b0['micro_f1']:.6f} / {b1['micro_f1']:.6f}",
        f"- B0/B1 schema-valid: {b0['schema_valid_rate']:.6f} / {b1['schema_valid_rate']:.6f}",
        f"- B0/B1 hallucination-keyword rate: {b0['hallucination_keyword_rate']:.6f} / {b1['hallucination_keyword_rate']:.6f}",
        f"- B1/B0 input-token ratio: {report['token_cost_ratio_b1_over_b0']['input']:.3f}",
        f"- B1/B0 output-token ratio: {report['token_cost_ratio_b1_over_b0']['output']:.3f}",
        f"- Max-token outputs B0/B1: {b0['max_token_output_count']} / {b1['max_token_output_count']}",
        f"- Provisional decision: `{report['provisional_decision']}`",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=Path("reports/baselines/qwen2_5_7b_dev/v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/baselines/qwen2_5_7b_dev/v1/diagnostics"))
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        paths = {
            "reference": args.baseline_dir / "dev_teacher_reference.jsonl",
            "b0": args.baseline_dir / "b0_simple_predictions.jsonl",
            "b1": args.baseline_dir / "b1_structured_v3_predictions.jsonl",
        }
        actual_hashes = {name: file_sha256(path) for name, path in paths.items()}
        if actual_hashes != EXPECTED_HASHES:
            raise ValueError(f"Baseline input hash mismatch: {actual_hashes}")
        manifest = json.loads((args.baseline_dir / "run_manifest.json").read_text(encoding="utf-8"))
        report, paired_rows = analyze(
            read_jsonl(paths["reference"]),
            read_jsonl(paths["b0"]),
            read_jsonl(paths["b1"]),
            manifest["max_new_tokens"],
            args.bootstrap_iterations,
            args.seed,
        )
        report["input_hashes"] = actual_hashes
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_file = args.output_dir / "paired_diagnostics.json"
        markdown_file = args.output_dir / "paired_diagnostics.md"
        samples_file = args.output_dir / "paired_samples.jsonl"
        write_json_atomic(report_file, report)
        write_text_atomic(markdown_file, render_markdown(report))
        write_jsonl_atomic(samples_file, paired_rows)
        log_diagnostics = {
            variant: {
                "schema_issue_counts": details["schema_issue_counts"],
                "max_token_output_count": details["max_token_output_count"],
                "top_false_positive": details["top_false_positive"][:10],
                "top_false_negative": details["top_false_negative"][:10],
                "top_hallucinated": details["top_hallucinated"][:10],
            }
            for variant, details in report["variants"].items()
        }
        emit(
            "p1_baseline_diagnostics_complete",
            status="PASS",
            report_file=str(report_file),
            report_sha256=file_sha256(report_file),
            markdown_file=str(markdown_file),
            markdown_sha256=file_sha256(markdown_file),
            samples_file=str(samples_file),
            samples_sha256=file_sha256(samples_file),
            paired=report["paired"],
            token_cost_ratio_b1_over_b0=report["token_cost_ratio_b1_over_b0"],
            checks=report["checks"],
            variant_diagnostics=log_diagnostics,
            provisional_decision=report["provisional_decision"],
        )
        return 0
    except Exception as exc:
        emit("p1_baseline_diagnostics_complete", status="FAIL", error_type=type(exc).__name__, error=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
