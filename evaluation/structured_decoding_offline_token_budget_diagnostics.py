"""Gate for deterministic offline vLLM structured decoding at 512 vs 768 tokens."""
import argparse
import json
from pathlib import Path
from typing import Any, Dict

from evaluation.keyword_evaluator import (
    file_sha256,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from evaluation.structured_decoding_inference import (
    CONSTRAINED_VARIANT as ORIGINAL_CONSTRAINED_VARIANT,
    CONTROL_VARIANT as ORIGINAL_CONTROL_VARIANT,
    REFERENCE_SHA256,
    emit,
)
from evaluation.structured_decoding_offline_token_budget_inference import (
    CANDIDATE_VARIANT,
    EXPERIMENT_VERSION,
    REPLAY_VARIANT,
)
from evaluation.structured_decoding_token_budget_diagnostics import analyze
from evaluation.structured_decoding_token_budget_inference import (
    ORIGINAL_CONSTRAINED_SHA256,
    ORIGINAL_CONTROL_SHA256,
)


def render_markdown(report: Dict[str, Any]) -> str:
    replay = report["metrics"][REPLAY_VARIANT]
    candidate = report["metrics"][CANDIDATE_VARIANT]
    recovery = report["original_truncation_recovery"]
    stability = report["nontruncated_exact_replay"]
    return "\n".join([
        "# Deterministic Offline Structured Decoding 512→768 Gate",
        "",
        f"- Samples: {report['sample_count']}",
        f"- Replay/candidate Macro-F1: {replay['macro_f1']:.6f} / {candidate['macro_f1']:.6f}",
        f"- Replay/candidate schema-valid: {replay['schema_valid_rate']:.6f} / {candidate['schema_valid_rate']:.6f}",
        f"- Replay/candidate runtime errors: {replay['runtime_error_count']} / {candidate['runtime_error_count']}",
        f"- Original truncations rescued: {recovery['rescued']} / {recovery['original_count']}",
        f"- Nontruncated raw-response exact rate: {stability['rate']:.6f}",
        f"- Replay fidelity pass: `{report['replay_fidelity_pass']}`",
        f"- Candidate pass: `{report['candidate_pass']}`",
        f"- Decision: `{report['decision']}`",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prior-dir", type=Path,
        default=Path("reports/baselines/qwen2_5_7b_dev/v5_structured_decoding"),
    )
    parser.add_argument(
        "--experiment-dir", type=Path,
        default=Path("reports/baselines/qwen2_5_7b_dev/v7_offline_deterministic_768"),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        paths = {
            "reference": args.experiment_dir / "dev_teacher_reference.jsonl",
            "prior_control": args.prior_dir
            / f"{ORIGINAL_CONTROL_VARIANT}_predictions.jsonl",
            "prior_constrained": args.prior_dir
            / f"{ORIGINAL_CONSTRAINED_VARIANT}_predictions.jsonl",
            "replay": args.experiment_dir / f"{REPLAY_VARIANT}_predictions.jsonl",
            "candidate": args.experiment_dir / f"{CANDIDATE_VARIANT}_predictions.jsonl",
        }
        expected_hashes = {
            "reference": REFERENCE_SHA256,
            "prior_control": ORIGINAL_CONTROL_SHA256,
            "prior_constrained": ORIGINAL_CONSTRAINED_SHA256,
        }
        for name, expected in expected_hashes.items():
            if file_sha256(paths[name]) != expected:
                raise ValueError(f"{name} hash mismatch")

        manifest = json.loads(
            (args.experiment_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        frozen_manifest = {
            "experiment_version": EXPERIMENT_VERSION,
            "base_prompt_variant": "b1_structured_v3",
            "response_format": "same_json_schema_both_arms",
            "temperature": 0,
            "seed": 42,
            "max_input_tokens": 4096,
            "max_model_len": 4864,
            "dtype": "bfloat16",
            "vllm_enable_v1_multiprocessing": "0",
            "inference_mode": "offline",
            "single_variable": "max_tokens_512_to_768",
            "arm_max_tokens": {
                REPLAY_VARIANT: 512,
                CANDIDATE_VARIANT: 768,
            },
        }
        drift = {
            name: [manifest.get(name), expected]
            for name, expected in frozen_manifest.items()
            if manifest.get(name) != expected
        }
        if drift:
            raise ValueError(f"Frozen offline configuration drift: {drift}")

        report, paired_rows = analyze(
            read_jsonl(paths["reference"]),
            read_jsonl(paths["prior_control"]),
            read_jsonl(paths["prior_constrained"]),
            read_jsonl(paths["replay"]),
            read_jsonl(paths["candidate"]),
            args.bootstrap_iterations,
            args.seed,
            replay_variant=REPLAY_VARIANT,
            candidate_variant=CANDIDATE_VARIANT,
        )
        report["report_version"] = EXPERIMENT_VERSION
        report["label"] = (
            "teacher_dev_offline_deterministic_token_budget_not_human_gold_test"
        )
        if report["decision"] == "ADOPT_JSON_SCHEMA_768":
            report["decision"] = "ADOPT_OFFLINE_JSON_SCHEMA_768"
        report["input_hashes"] = {name: file_sha256(path) for name, path in paths.items()}

        output_dir = args.experiment_dir / "diagnostics"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / "offline_structured_decoding_token_budget_gate.json"
        markdown_file = output_dir / "offline_structured_decoding_token_budget_gate.md"
        paired_file = output_dir / "offline_structured_decoding_token_budget_paired.jsonl"
        write_json_atomic(report_file, report)
        write_text_atomic(markdown_file, render_markdown(report))
        write_jsonl_atomic(paired_file, paired_rows)
        emit(
            "p1_structured_decoding_offline_token_budget_gate_complete",
            status="PASS",
            decision=report["decision"],
            replay_fidelity_pass=report["replay_fidelity_pass"],
            candidate_pass=report["candidate_pass"],
            replay_fidelity_checks=report["replay_fidelity_checks"],
            candidate_checks=report["candidate_checks"],
            nontruncated_exact_replay=report["nontruncated_exact_replay"],
            recovery=report["original_truncation_recovery"],
            paired_candidate_minus_replay=report["paired_candidate_minus_replay"],
            paired_candidate_minus_prior_control=report[
                "paired_candidate_minus_prior_control"
            ],
            metrics={
                REPLAY_VARIANT: report["metrics"][REPLAY_VARIANT],
                CANDIDATE_VARIANT: report["metrics"][CANDIDATE_VARIANT],
            },
            report_file=str(report_file),
            report_sha256=file_sha256(report_file),
        )
        return 0
    except Exception as exc:
        emit(
            "p1_structured_decoding_offline_token_budget_gate_complete",
            status="FAIL",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
