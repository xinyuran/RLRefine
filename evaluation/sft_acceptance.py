"""Pre-registered teacher-dev acceptance decision for the first SFT candidate."""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import yaml

from evaluation.baseline_diagnostics import analyze
from evaluation.keyword_evaluator import (
    file_sha256,
    read_jsonl,
    write_json_atomic,
    write_text_atomic,
)


BASELINE_REFERENCE_SHA256 = "2c4cd0e5eaf14dc4ade0ac857e70d5669d9ea4746608c46c6bf3df6df3d83dfe"
BASELINE_PREDICTION_SHA256 = "deec1c9138efa09910c6cd202f92d2c1bf3979b0a9f09db197bef7a5f4d59abc"


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True), flush=True)


def decide(
    reference_rows: Sequence[Dict[str, Any]],
    baseline_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
    gates: Dict[str, Any],
    bootstrap_iterations: int = 10000,
    seed: int = 42,
) -> Tuple[Dict[str, Any], Sequence[Dict[str, Any]]]:
    diagnostics, paired_rows = analyze(
        reference_rows,
        baseline_rows,
        candidate_rows,
        max_new_tokens=512,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    candidate = diagnostics["variants"]["b1_structured_v3"]
    paired = diagnostics["paired"]
    checks = {
        "paired_macro_f1_delta_min": (
            paired["macro_f1_delta_b1_minus_b0"] >= gates["paired_macro_f1_delta_min"]
        ),
        "paired_macro_f1_bootstrap_95_ci_low_gt": (
            paired["bootstrap_95_ci"][0] > gates["paired_macro_f1_bootstrap_95_ci_low_gt"]
        ),
        "micro_f1_min": candidate["micro_f1"] >= gates["micro_f1_min"],
        "schema_valid_rate_min": (
            candidate["schema_valid_rate"] >= gates["schema_valid_rate_min"]
        ),
        "hallucination_keyword_rate_max": (
            candidate["hallucination_keyword_rate"] <= gates["hallucination_keyword_rate_max"]
        ),
        "mean_output_tokens_max": (
            candidate["mean_output_tokens"] <= gates["mean_output_tokens_max"]
        ),
        "max_token_output_count_max": (
            candidate["max_token_output_count"] <= gates["max_token_output_count_max"]
        ),
    }
    accepted = all(checks.values())
    return {
        "report_version": "sft-teacher-dev-acceptance-v1",
        "label": "teacher_dev_selection_not_human_gold_test",
        "sample_count": len(reference_rows),
        "bootstrap_iterations": bootstrap_iterations,
        "seed": seed,
        "paired": {
            "macro_f1_delta_candidate_minus_base_b1": paired["macro_f1_delta_b1_minus_b0"],
            "bootstrap_95_ci": paired["bootstrap_95_ci"],
            "outcomes": {
                key.replace("b1_win", "candidate_win").replace("b0_win", "base_b1_win"): value
                for key, value in paired["outcomes"].items()
            },
        },
        "baseline": diagnostics["variants"]["b0_simple"],
        "candidate": candidate,
        "gates": gates,
        "checks": checks,
        "decision": "ACCEPT" if accepted else "REJECT",
    }, paired_rows


def render_markdown(report: Dict[str, Any]) -> str:
    paired = report["paired"]
    return "\n".join([
        "# SFT Teacher-Dev Acceptance",
        "",
        f"- Decision: **{report['decision']}**",
        f"- Samples: {report['sample_count']}",
        f"- Candidate minus Base B1 macro-F1: {paired['macro_f1_delta_candidate_minus_base_b1']:.6f}",
        f"- Paired bootstrap 95% CI: [{paired['bootstrap_95_ci'][0]:.6f}, {paired['bootstrap_95_ci'][1]:.6f}]",
        f"- Candidate micro/macro-F1: {report['candidate']['micro_f1']:.6f} / {report['candidate']['macro_f1']:.6f}",
        f"- Candidate schema-valid: {report['candidate']['schema_valid_rate']:.6f}",
        f"- Candidate hallucination-keyword rate: {report['candidate']['hallucination_keyword_rate']:.6f}",
        f"- Candidate mean output tokens: {report['candidate']['mean_output_tokens']:.3f}",
        f"- Gate checks: `{json.dumps(report['checks'], sort_keys=True)}`",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/p1_sft_lora_v1.yaml"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("reports/baselines/qwen2_5_7b_dev/v1"))
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        acceptance = config["acceptance"]
        paths = {
            "reference": args.baseline_dir / "dev_teacher_reference.jsonl",
            "baseline": args.baseline_dir / "b1_structured_v3_predictions.jsonl",
            "candidate_reference": args.candidate_dir / "dev_teacher_reference.jsonl",
            "candidate": args.candidate_dir / "b1_structured_v3_predictions.jsonl",
        }
        actual_hashes = {name: file_sha256(path) for name, path in paths.items()}
        if actual_hashes["reference"] != BASELINE_REFERENCE_SHA256:
            raise ValueError("Frozen teacher-dev reference hash mismatch")
        if actual_hashes["baseline"] != BASELINE_PREDICTION_SHA256:
            raise ValueError("Frozen Base B1 prediction hash mismatch")
        if actual_hashes["candidate_reference"] != actual_hashes["reference"]:
            raise ValueError("Candidate and baseline teacher-dev references differ")
        if acceptance["baseline_prediction_sha256"] != actual_hashes["baseline"]:
            raise ValueError("Experiment config baseline prediction hash mismatch")
        report, _ = decide(
            read_jsonl(paths["reference"]),
            read_jsonl(paths["baseline"]),
            read_jsonl(paths["candidate"]),
            acceptance["gates"],
            args.bootstrap_iterations,
            args.seed,
        )
        report["input_hashes"] = actual_hashes
        report["acceptance_policy"] = acceptance["policy"]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_file = args.output_dir / "sft_teacher_dev_acceptance.json"
        markdown_file = args.output_dir / "sft_teacher_dev_acceptance.md"
        write_json_atomic(report_file, report)
        write_text_atomic(markdown_file, render_markdown(report))
        emit(
            "p1_sft_teacher_dev_acceptance_complete",
            status="PASS",
            decision=report["decision"],
            report_file=str(report_file),
            report_sha256=file_sha256(report_file),
            markdown_file=str(markdown_file),
            markdown_sha256=file_sha256(markdown_file),
            paired=report["paired"],
            candidate=report["candidate"],
            checks=report["checks"],
        )
        return 0
    except Exception as exc:
        emit("p1_sft_teacher_dev_acceptance_complete", status="FAIL", error_type=type(exc).__name__, error=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
