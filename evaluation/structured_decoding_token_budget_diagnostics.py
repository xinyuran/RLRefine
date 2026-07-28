"""Pre-registered gate for increasing structured decoding from 512 to 768 tokens."""
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
    file_sha256,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from evaluation.structured_decoding_diagnostics import _metrics
from evaluation.structured_decoding_inference import (
    CONSTRAINED_VARIANT as ORIGINAL_CONSTRAINED_VARIANT,
    CONTROL_VARIANT as ORIGINAL_CONTROL_VARIANT,
    REFERENCE_SHA256,
    emit,
)
from evaluation.structured_decoding_token_budget_inference import (
    CANDIDATE_VARIANT,
    EXPERIMENT_VERSION,
    ORIGINAL_CONSTRAINED_SHA256,
    ORIGINAL_CONTROL_SHA256,
    REPLAY_VARIANT,
)


THRESHOLDS = {
    "replay_max_macro_f1_abs_delta": 0.01,
    "replay_max_micro_f1_abs_delta": 0.01,
    "replay_max_schema_valid_rate_abs_delta": 0.01,
    "replay_max_hallucination_rate_abs_delta": 0.005,
    "min_nontruncated_exact_replay_rate": 0.99,
    "candidate_vs_replay_macro_f1_noninferiority_margin": 0.0,
    "candidate_vs_control_macro_f1_noninferiority_margin": 0.03,
    "candidate_max_micro_f1_drop_vs_control": 0.02,
    "candidate_min_schema_valid_rate": 0.99,
    "candidate_max_hallucination_rate_increase_vs_control": 0.01,
    "candidate_max_input_token_ratio_vs_control": 1.01,
    "candidate_max_output_token_ratio_vs_control": 1.05,
}


def _paired_deltas(
    reference: Dict[str, Dict[str, Any]],
    left: Dict[str, Dict[str, Any]],
    right: Dict[str, Dict[str, Any]],
    bootstrap_iterations: int,
    seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    deltas: List[float] = []
    outcomes = {"right_win": 0, "left_win": 0, "tie": 0}
    rows = []
    for sample_id in sorted(reference):
        source = reference[sample_id]["normalized_text"]
        gold = set(reference[sample_id]["gold"]["keywords"])
        left_f1 = _sample_f1(_prediction_view(left[sample_id], source)["keywords"], gold)
        right_f1 = _sample_f1(_prediction_view(right[sample_id], source)["keywords"], gold)
        delta = right_f1 - left_f1
        deltas.append(delta)
        outcome = "right_win" if delta > 0 else "left_win" if delta < 0 else "tie"
        outcomes[outcome] += 1
        rows.append({
            "sample_id": sample_id,
            "left_f1": left_f1,
            "right_f1": right_f1,
            "f1_delta_right_minus_left": delta,
            "outcome": outcome,
        })
    ci_low, ci_high = _bootstrap_mean_ci(deltas, bootstrap_iterations, seed)
    return {
        "macro_f1_delta_right_minus_left": sum(deltas) / len(deltas),
        "bootstrap_95_ci": [ci_low, ci_high],
        "outcomes": outcomes,
    }, rows


def analyze(
    reference_rows: Sequence[Dict[str, Any]],
    prior_control_rows: Sequence[Dict[str, Any]],
    prior_constrained_rows: Sequence[Dict[str, Any]],
    replay_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
    bootstrap_iterations: int = 10000,
    seed: int = 42,
    replay_variant: str = REPLAY_VARIANT,
    candidate_variant: str = CANDIDATE_VARIANT,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    reference = _index_unique(reference_rows, "reference")
    indexed = {
        ORIGINAL_CONTROL_VARIANT: _index_unique(prior_control_rows, ORIGINAL_CONTROL_VARIANT),
        ORIGINAL_CONSTRAINED_VARIANT: _index_unique(
            prior_constrained_rows, ORIGINAL_CONSTRAINED_VARIANT
        ),
        replay_variant: _index_unique(replay_rows, replay_variant),
        candidate_variant: _index_unique(candidate_rows, candidate_variant),
    }
    if not reference or any(set(rows) != set(reference) for rows in indexed.values()):
        raise ValueError("Reference/prior/replay/candidate sample coverage mismatch")
    for variant, rows in (
        (ORIGINAL_CONTROL_VARIANT, prior_control_rows),
        (ORIGINAL_CONSTRAINED_VARIANT, prior_constrained_rows),
        (replay_variant, replay_rows),
        (candidate_variant, candidate_rows),
    ):
        if any(row.get("prompt_variant") != variant for row in rows):
            raise ValueError(f"Prediction rows do not match variant {variant}")

    metrics = {
        ORIGINAL_CONTROL_VARIANT: _metrics(
            reference_rows, prior_control_rows, ORIGINAL_CONTROL_VARIANT, 512
        ),
        ORIGINAL_CONSTRAINED_VARIANT: _metrics(
            reference_rows, prior_constrained_rows, ORIGINAL_CONSTRAINED_VARIANT, 512
        ),
        replay_variant: _metrics(reference_rows, replay_rows, replay_variant, 512),
        candidate_variant: _metrics(
            reference_rows, candidate_rows, candidate_variant, 768
        ),
    }
    prior = metrics[ORIGINAL_CONSTRAINED_VARIANT]
    replay = metrics[replay_variant]
    control = metrics[ORIGINAL_CONTROL_VARIANT]
    candidate = metrics[candidate_variant]

    replay_checks = {
        "replay_macro_f1_matches_prior":
            abs(replay["macro_f1"] - prior["macro_f1"])
            <= THRESHOLDS["replay_max_macro_f1_abs_delta"],
        "replay_micro_f1_matches_prior":
            abs(replay["micro_f1"] - prior["micro_f1"])
            <= THRESHOLDS["replay_max_micro_f1_abs_delta"],
        "replay_schema_rate_matches_prior":
            abs(replay["schema_valid_rate"] - prior["schema_valid_rate"])
            <= THRESHOLDS["replay_max_schema_valid_rate_abs_delta"],
        "replay_hallucination_matches_prior":
            abs(replay["hallucination_keyword_rate"] - prior["hallucination_keyword_rate"])
            <= THRESHOLDS["replay_max_hallucination_rate_abs_delta"],
    }

    stopped_replay_ids = [
        sample_id
        for sample_id, row in indexed[replay_variant].items()
        if row.get("finish_reason") == "stop"
    ]
    exact_stopped = sum(
        indexed[replay_variant][sample_id].get("raw_response")
        == indexed[candidate_variant][sample_id].get("raw_response")
        for sample_id in stopped_replay_ids
    )
    nontruncated_exact_rate = (
        exact_stopped / len(stopped_replay_ids) if stopped_replay_ids else 0.0
    )
    original_truncated_ids = sorted(
        sample_id
        for sample_id, row in indexed[ORIGINAL_CONSTRAINED_VARIANT].items()
        if row.get("finish_reason") == "length" or row.get("output_tokens", 0) >= 512
    )
    rescued_original_truncations = sum(
        indexed[candidate_variant][sample_id].get("status") == "success"
        and indexed[candidate_variant][sample_id].get("finish_reason") == "stop"
        for sample_id in original_truncated_ids
    )

    replay_pair, paired_rows = _paired_deltas(
        reference,
        indexed[replay_variant],
        indexed[candidate_variant],
        bootstrap_iterations,
        seed,
    )
    control_pair, _ = _paired_deltas(
        reference,
        indexed[ORIGINAL_CONTROL_VARIANT],
        indexed[candidate_variant],
        bootstrap_iterations,
        seed + 1,
    )
    input_ratio = candidate["mean_input_tokens"] / control["mean_input_tokens"]
    output_ratio = candidate["mean_output_tokens"] / control["mean_output_tokens"]
    length_finish_count = sum(
        row.get("finish_reason") == "length" for row in candidate_rows
    )
    candidate_checks = {
        "candidate_nontruncated_outputs_are_causally_stable":
            nontruncated_exact_rate
            >= THRESHOLDS["min_nontruncated_exact_replay_rate"],
        "candidate_macro_f1_noninferior_to_replay":
            replay_pair["bootstrap_95_ci"][0]
            >= -THRESHOLDS["candidate_vs_replay_macro_f1_noninferiority_margin"],
        "candidate_macro_f1_noninferior_to_control":
            control_pair["bootstrap_95_ci"][0]
            >= -THRESHOLDS["candidate_vs_control_macro_f1_noninferiority_margin"],
        "candidate_micro_f1_within_control_tolerance":
            candidate["micro_f1"]
            >= control["micro_f1"]
            - THRESHOLDS["candidate_max_micro_f1_drop_vs_control"],
        "candidate_schema_valid_rate_at_least_99_percent":
            candidate["schema_valid_rate"]
            >= THRESHOLDS["candidate_min_schema_valid_rate"],
        "candidate_hallucination_within_control_tolerance":
            candidate["hallucination_keyword_rate"]
            <= control["hallucination_keyword_rate"]
            + THRESHOLDS["candidate_max_hallucination_rate_increase_vs_control"],
        "candidate_input_tokens_within_one_percent":
            input_ratio <= THRESHOLDS["candidate_max_input_token_ratio_vs_control"],
        "candidate_output_tokens_within_five_percent":
            output_ratio <= THRESHOLDS["candidate_max_output_token_ratio_vs_control"],
        "candidate_no_runtime_errors": candidate["runtime_error_count"] == 0,
        "candidate_no_length_finishes": length_finish_count == 0,
        "candidate_no_768_token_outputs": candidate["max_token_output_count"] == 0,
        "candidate_rescues_all_original_truncations":
            rescued_original_truncations == len(original_truncated_ids),
    }
    replay_pass = all(replay_checks.values())
    candidate_pass = all(candidate_checks.values())
    decision = (
        "ADOPT_JSON_SCHEMA_768"
        if replay_pass and candidate_pass
        else "INVALID_512_REPLAY"
        if not replay_pass
        else "RETAIN_UNCONSTRAINED_B1"
    )
    for row in paired_rows:
        sample_id = row["sample_id"]
        row.update({
            "replay_finish_reason": indexed[replay_variant][sample_id].get("finish_reason"),
            "candidate_finish_reason": indexed[candidate_variant][sample_id].get("finish_reason"),
            "raw_response_exact": (
                indexed[replay_variant][sample_id].get("raw_response")
                == indexed[candidate_variant][sample_id].get("raw_response")
            ),
        })
    report = {
        "report_version": EXPERIMENT_VERSION,
        "label": "teacher_dev_structured_decoding_token_budget_not_human_gold_test",
        "sample_count": len(reference),
        "bootstrap_iterations": bootstrap_iterations,
        "seed": seed,
        "thresholds": THRESHOLDS,
        "metrics": metrics,
        "replay_variant": replay_variant,
        "candidate_variant": candidate_variant,
        "replay_fidelity_checks": replay_checks,
        "replay_fidelity_pass": replay_pass,
        "candidate_checks": candidate_checks,
        "candidate_pass": candidate_pass,
        "nontruncated_exact_replay": {
            "eligible": len(stopped_replay_ids),
            "exact": exact_stopped,
            "rate": nontruncated_exact_rate,
        },
        "original_truncation_recovery": {
            "original_count": len(original_truncated_ids),
            "rescued": rescued_original_truncations,
            "sample_ids": original_truncated_ids,
        },
        "paired_candidate_minus_replay": replay_pair,
        "paired_candidate_minus_prior_control": control_pair,
        "cost_ratios_candidate_over_prior_control": {
            "input_tokens": input_ratio,
            "output_tokens": output_ratio,
            "mean_request_seconds": (
                candidate["mean_request_seconds"] / control["mean_request_seconds"]
                if control["mean_request_seconds"] else None
            ),
        },
        "candidate_length_finish_count": length_finish_count,
        "decision": decision,
    }
    return report, paired_rows


def render_markdown(report: Dict[str, Any]) -> str:
    replay = report["metrics"][REPLAY_VARIANT]
    candidate = report["metrics"][CANDIDATE_VARIANT]
    recovery = report["original_truncation_recovery"]
    return "\n".join([
        "# Structured Decoding 512→768 Token-Budget Gate",
        "",
        f"- Samples: {report['sample_count']}",
        f"- Replay/candidate Macro-F1: {replay['macro_f1']:.6f} / {candidate['macro_f1']:.6f}",
        f"- Replay/candidate schema-valid: {replay['schema_valid_rate']:.6f} / {candidate['schema_valid_rate']:.6f}",
        f"- Replay/candidate runtime errors: {replay['runtime_error_count']} / {candidate['runtime_error_count']}",
        f"- Original truncations rescued: {recovery['rescued']} / {recovery['original_count']}",
        f"- Nontruncated exact replay rate: {report['nontruncated_exact_replay']['rate']:.6f}",
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
        default=Path("reports/baselines/qwen2_5_7b_dev/v6_structured_decoding_768"),
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
            "workers": 4,
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
            raise ValueError(f"Frozen token-budget configuration drift: {drift}")
        report, paired_rows = analyze(
            read_jsonl(paths["reference"]),
            read_jsonl(paths["prior_control"]),
            read_jsonl(paths["prior_constrained"]),
            read_jsonl(paths["replay"]),
            read_jsonl(paths["candidate"]),
            args.bootstrap_iterations,
            args.seed,
        )
        report["input_hashes"] = {name: file_sha256(path) for name, path in paths.items()}
        output_dir = args.experiment_dir / "diagnostics"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / "structured_decoding_token_budget_gate.json"
        markdown_file = output_dir / "structured_decoding_token_budget_gate.md"
        paired_file = output_dir / "structured_decoding_token_budget_paired.jsonl"
        write_json_atomic(report_file, report)
        write_text_atomic(markdown_file, render_markdown(report))
        write_jsonl_atomic(paired_file, paired_rows)
        emit(
            "p1_structured_decoding_token_budget_gate_complete",
            status="PASS",
            decision=report["decision"],
            replay_fidelity_pass=report["replay_fidelity_pass"],
            candidate_pass=report["candidate_pass"],
            replay_fidelity_checks=report["replay_fidelity_checks"],
            candidate_checks=report["candidate_checks"],
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
            "p1_structured_decoding_token_budget_gate_complete",
            status="FAIL",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
