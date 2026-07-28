"""Pre-registered decision gate for the vLLM JSON-Schema decoding experiment."""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from evaluation.baseline_diagnostics import (
    _bootstrap_mean_ci,
    _index_unique,
    _prediction_view,
    _sample_f1,
)
from evaluation.keyword_evaluator import (
    evaluate_records,
    file_sha256,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from evaluation.structured_decoding_inference import (
    CONSTRAINED_VARIANT,
    CONTROL_VARIANT,
    EXPERIMENT_VERSION,
    REFERENCE_SHA256,
    emit,
)


FROZEN_B1_SHA256 = "deec1c9138efa09910c6cd202f92d2c1bf3979b0a9f09db197bef7a5f4d59abc"
FROZEN_B1_VARIANT = "b1_structured_v3"
THRESHOLDS = {
    "backend_max_macro_f1_abs_delta": 0.03,
    "backend_max_micro_f1_abs_delta": 0.03,
    "backend_max_schema_valid_rate_drop": 0.03,
    "backend_max_hallucination_rate_increase": 0.015,
    "constrained_macro_f1_noninferiority_margin": 0.03,
    "constrained_max_micro_f1_drop": 0.02,
    "constrained_min_schema_valid_rate": 0.99,
    "constrained_max_hallucination_rate_increase": 0.01,
    "constrained_max_input_token_ratio": 1.01,
    "constrained_max_output_token_ratio": 1.05,
}


def _metrics(
    reference_rows: Sequence[Dict[str, Any]],
    predictions: Sequence[Dict[str, Any]],
    variant: str,
    max_tokens: int,
) -> Dict[str, Any]:
    report = evaluate_records(reference_rows, predictions, variant)[0]
    count = len(predictions)
    runtime_error_count = sum(row.get("status") != "success" for row in predictions)
    return {
        "macro_f1": report["macro"]["f1"],
        "micro_f1": report["micro"]["f1"],
        "schema_valid_rate": report["schema_valid_rate"],
        "hallucination_keyword_rate": report["hallucination"]["keyword_rate"],
        "exact_set_match_rate": report["exact_set_match_rate"],
        "mean_input_tokens": sum(row["input_tokens"] for row in predictions) / count,
        "mean_output_tokens": sum(row["output_tokens"] for row in predictions) / count,
        "mean_request_seconds": sum(row.get("request_seconds", 0) for row in predictions) / count,
        "max_token_output_count": sum(row["output_tokens"] >= max_tokens for row in predictions),
        "runtime_error_count": runtime_error_count,
        "evaluation_difference_sample_count": report["error_sample_count"],
        "status_counts": report["status_counts"],
    }


def analyze(
    reference_rows: Sequence[Dict[str, Any]],
    frozen_rows: Sequence[Dict[str, Any]],
    control_rows: Sequence[Dict[str, Any]],
    constrained_rows: Sequence[Dict[str, Any]],
    max_tokens: int = 512,
    bootstrap_iterations: int = 10000,
    seed: int = 42,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    reference = _index_unique(reference_rows, "reference")
    arms = {
        FROZEN_B1_VARIANT: _index_unique(frozen_rows, FROZEN_B1_VARIANT),
        CONTROL_VARIANT: _index_unique(control_rows, CONTROL_VARIANT),
        CONSTRAINED_VARIANT: _index_unique(constrained_rows, CONSTRAINED_VARIANT),
    }
    if not reference or any(set(rows) != set(reference) for rows in arms.values()):
        raise ValueError("Reference/frozen/control/constrained sample coverage mismatch")
    for variant, rows in ((CONTROL_VARIANT, control_rows), (CONSTRAINED_VARIANT, constrained_rows)):
        if any(row.get("prompt_variant") != variant for row in rows):
            raise ValueError(f"Prediction rows do not match variant {variant}")

    metrics = {
        variant: _metrics(reference_rows, rows, variant, max_tokens)
        for variant, rows in (
            (FROZEN_B1_VARIANT, frozen_rows),
            (CONTROL_VARIANT, control_rows),
            (CONSTRAINED_VARIANT, constrained_rows),
        )
    }
    paired_rows: List[Dict[str, Any]] = []
    deltas: List[float] = []
    outcomes = {"constrained_win": 0, "control_win": 0, "tie": 0}
    for sample_id in sorted(reference):
        source = reference[sample_id]["normalized_text"]
        gold = set(reference[sample_id]["gold"]["keywords"])
        control_view = _prediction_view(arms[CONTROL_VARIANT][sample_id], source)
        constrained_view = _prediction_view(arms[CONSTRAINED_VARIANT][sample_id], source)
        control_f1 = _sample_f1(control_view["keywords"], gold)
        constrained_f1 = _sample_f1(constrained_view["keywords"], gold)
        delta = constrained_f1 - control_f1
        deltas.append(delta)
        outcome = "constrained_win" if delta > 0 else "control_win" if delta < 0 else "tie"
        outcomes[outcome] += 1
        paired_rows.append({
            "sample_id": sample_id,
            "control_f1": control_f1,
            "constrained_f1": constrained_f1,
            "f1_delta_constrained_minus_control": delta,
            "outcome": outcome,
            "control_schema_issue": control_view["schema_issue"],
            "constrained_schema_issue": constrained_view["schema_issue"],
            "control_output_tokens": control_view["output_tokens"],
            "constrained_output_tokens": constrained_view["output_tokens"],
        })

    ci_low, ci_high = _bootstrap_mean_ci(deltas, bootstrap_iterations, seed)
    frozen = metrics[FROZEN_B1_VARIANT]
    control = metrics[CONTROL_VARIANT]
    constrained = metrics[CONSTRAINED_VARIANT]
    backend_checks = {
        "control_macro_f1_reproduces_frozen_b1":
            abs(control["macro_f1"] - frozen["macro_f1"])
            <= THRESHOLDS["backend_max_macro_f1_abs_delta"],
        "control_micro_f1_reproduces_frozen_b1":
            abs(control["micro_f1"] - frozen["micro_f1"])
            <= THRESHOLDS["backend_max_micro_f1_abs_delta"],
        "control_schema_rate_reproduces_frozen_b1":
            control["schema_valid_rate"]
            >= frozen["schema_valid_rate"] - THRESHOLDS["backend_max_schema_valid_rate_drop"],
        "control_hallucination_reproduces_frozen_b1":
            control["hallucination_keyword_rate"]
            <= frozen["hallucination_keyword_rate"]
            + THRESHOLDS["backend_max_hallucination_rate_increase"],
    }
    input_ratio = constrained["mean_input_tokens"] / control["mean_input_tokens"]
    output_ratio = constrained["mean_output_tokens"] / control["mean_output_tokens"]
    structured_checks = {
        "constrained_macro_f1_noninferior":
            ci_low >= -THRESHOLDS["constrained_macro_f1_noninferiority_margin"],
        "constrained_micro_f1_within_tolerance":
            constrained["micro_f1"]
            >= control["micro_f1"] - THRESHOLDS["constrained_max_micro_f1_drop"],
        "constrained_schema_valid_rate_at_least_99_percent":
            constrained["schema_valid_rate"]
            >= THRESHOLDS["constrained_min_schema_valid_rate"],
        "constrained_hallucination_within_tolerance":
            constrained["hallucination_keyword_rate"]
            <= control["hallucination_keyword_rate"]
            + THRESHOLDS["constrained_max_hallucination_rate_increase"],
        "constrained_input_tokens_within_one_percent":
            input_ratio <= THRESHOLDS["constrained_max_input_token_ratio"],
        "constrained_output_tokens_within_five_percent":
            output_ratio <= THRESHOLDS["constrained_max_output_token_ratio"],
        "constrained_no_max_token_outputs": constrained["max_token_output_count"] == 0,
        "constrained_no_runtime_errors": constrained["runtime_error_count"] == 0,
    }
    backend_pass = all(backend_checks.values())
    structured_pass = all(structured_checks.values())
    decision = (
        "ADOPT_JSON_SCHEMA_DECODING"
        if backend_pass and structured_pass
        else "INVALID_BACKEND_COMPARISON"
        if not backend_pass
        else "RETAIN_UNCONSTRAINED_B1"
    )
    report = {
        "report_version": EXPERIMENT_VERSION,
        "label": "teacher_dev_structured_decoding_not_human_gold_test",
        "sample_count": len(reference),
        "bootstrap_iterations": bootstrap_iterations,
        "seed": seed,
        "thresholds": THRESHOLDS,
        "metrics": metrics,
        "paired": {
            "macro_f1_delta_constrained_minus_control": sum(deltas) / len(deltas),
            "bootstrap_95_ci": [ci_low, ci_high],
            "outcomes": outcomes,
        },
        "cost_ratios_constrained_over_control": {
            "input_tokens": input_ratio,
            "output_tokens": output_ratio,
            "mean_request_seconds": (
                constrained["mean_request_seconds"] / control["mean_request_seconds"]
                if control["mean_request_seconds"] else None
            ),
        },
        "backend_fidelity_checks": backend_checks,
        "structured_decoding_checks": structured_checks,
        "backend_fidelity_pass": backend_pass,
        "structured_decoding_pass": structured_pass,
        "decision": decision,
    }
    return report, paired_rows


def render_markdown(report: Dict[str, Any]) -> str:
    frozen = report["metrics"][FROZEN_B1_VARIANT]
    control = report["metrics"][CONTROL_VARIANT]
    constrained = report["metrics"][CONSTRAINED_VARIANT]
    paired = report["paired"]
    ratios = report["cost_ratios_constrained_over_control"]
    return "\n".join([
        "# B1 vLLM JSON-Schema Structured-Decoding Gate",
        "",
        f"- Label: `{report['label']}`",
        f"- Samples: {report['sample_count']}",
        f"- Frozen/control/constrained Macro-F1: {frozen['macro_f1']:.6f} / {control['macro_f1']:.6f} / {constrained['macro_f1']:.6f}",
        f"- Control/constrained Micro-F1: {control['micro_f1']:.6f} / {constrained['micro_f1']:.6f}",
        f"- Control/constrained schema-valid rate: {control['schema_valid_rate']:.6f} / {constrained['schema_valid_rate']:.6f}",
        f"- Control/constrained hallucination rate: {control['hallucination_keyword_rate']:.6f} / {constrained['hallucination_keyword_rate']:.6f}",
        f"- Paired Macro-F1 delta (constrained-control): {paired['macro_f1_delta_constrained_minus_control']:.6f}",
        f"- Paired bootstrap 95% CI: [{paired['bootstrap_95_ci'][0]:.6f}, {paired['bootstrap_95_ci'][1]:.6f}]",
        f"- Input/output token ratios: {ratios['input_tokens']:.3f} / {ratios['output_tokens']:.3f}",
        f"- Backend fidelity pass: `{report['backend_fidelity_pass']}`",
        f"- Structured decoding pass: `{report['structured_decoding_pass']}`",
        f"- Decision: `{report['decision']}`",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-dir", type=Path,
        default=Path("reports/baselines/qwen2_5_7b_dev/v1"),
    )
    parser.add_argument(
        "--experiment-dir", type=Path,
        default=Path("reports/baselines/qwen2_5_7b_dev/v5_structured_decoding"),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        paths = {
            "reference": args.experiment_dir / "dev_teacher_reference.jsonl",
            "frozen": args.baseline_dir / "b1_structured_v3_predictions.jsonl",
            "control": args.experiment_dir / f"{CONTROL_VARIANT}_predictions.jsonl",
            "constrained": args.experiment_dir / f"{CONSTRAINED_VARIANT}_predictions.jsonl",
        }
        if file_sha256(paths["reference"]) != REFERENCE_SHA256:
            raise ValueError("Teacher-dev reference hash mismatch")
        if file_sha256(paths["frozen"]) != FROZEN_B1_SHA256:
            raise ValueError("Frozen B1 prediction hash mismatch")
        manifest = json.loads(
            (args.experiment_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("experiment_version") != EXPERIMENT_VERSION:
            raise ValueError("Experiment manifest version mismatch")
        expected_manifest_fields = {
            "base_prompt_variant": "b1_structured_v3",
            "temperature": 0,
            "seed": 42,
            "max_input_tokens": 4096,
            "max_tokens": 512,
            "workers": 4,
            "single_variable": "response_format_json_schema",
        }
        drift = {
            name: [manifest.get(name), expected]
            for name, expected in expected_manifest_fields.items()
            if manifest.get(name) != expected
        }
        if drift:
            raise ValueError(f"Frozen structured-decoding configuration drift: {drift}")
        report, paired_rows = analyze(
            read_jsonl(paths["reference"]),
            read_jsonl(paths["frozen"]),
            read_jsonl(paths["control"]),
            read_jsonl(paths["constrained"]),
            max_tokens=manifest["max_tokens"],
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        report["input_hashes"] = {name: file_sha256(path) for name, path in paths.items()}
        output_dir = args.experiment_dir / "diagnostics"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / "structured_decoding_gate.json"
        markdown_file = output_dir / "structured_decoding_gate.md"
        paired_file = output_dir / "structured_decoding_paired_samples.jsonl"
        write_json_atomic(report_file, report)
        write_text_atomic(markdown_file, render_markdown(report))
        write_jsonl_atomic(paired_file, paired_rows)
        emit(
            "p1_structured_decoding_gate_complete",
            status="PASS",
            decision=report["decision"],
            backend_fidelity_pass=report["backend_fidelity_pass"],
            structured_decoding_pass=report["structured_decoding_pass"],
            paired=report["paired"],
            checks={
                **report["backend_fidelity_checks"],
                **report["structured_decoding_checks"],
            },
            report_file=str(report_file),
            report_sha256=file_sha256(report_file),
        )
        return 0
    except Exception as exc:
        emit(
            "p1_structured_decoding_gate_complete",
            status="FAIL",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
