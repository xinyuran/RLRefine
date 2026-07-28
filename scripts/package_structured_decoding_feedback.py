"""Create a compact, auditable handoff packet from the offline token-budget gate."""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from evaluation.keyword_evaluator import file_sha256, write_json_atomic, write_text_atomic


ADOPT_DECISION = "ADOPT_OFFLINE_JSON_SCHEMA_768"
INVALID_DECISION = "INVALID_512_REPLAY"
RETAIN_DECISION = "RETAIN_UNCONSTRAINED_B1"


def _failed(checks: Mapping[str, Any]) -> list[str]:
    return sorted(name for name, passed in checks.items() if passed is not True)


def build_feedback_packet(
    report: Mapping[str, Any],
    *,
    report_sha256: str,
    manifest_sha256: str,
) -> Dict[str, Any]:
    decision = report["decision"]
    if decision == ADOPT_DECISION:
        next_work_package = "WP2_OFFLINE_COLD_START_REPLICATION"
        authorization = (
            "replicate_offline_protocol_only; online_concurrent_serving_not_authorized"
        )
    elif decision == INVALID_DECISION:
        next_work_package = "WP1R_REPLAY_ROOT_CAUSE"
        authorization = "diagnostics_only; no_candidate_adoption_or_gold_evaluation"
    elif decision == RETAIN_DECISION:
        next_work_package = "WP3_CHALLENGE_SET_AND_ERROR_TAXONOMY"
        authorization = "retain_b1; stop_token_budget_iteration_on_teacher_dev"
    else:
        raise ValueError(f"Unsupported gate decision: {decision}")

    replay_variant = report["replay_variant"]
    candidate_variant = report["candidate_variant"]
    metrics = report["metrics"]
    return {
        "packet_version": "p1-structured-decoding-feedback-v1",
        "decision": decision,
        "next_work_package": next_work_package,
        "authorization": authorization,
        "failed_replay_checks": _failed(report["replay_fidelity_checks"]),
        "failed_candidate_checks": _failed(report["candidate_checks"]),
        "sample_count": report["sample_count"],
        "variants": {
            "replay": replay_variant,
            "candidate": candidate_variant,
        },
        "headline_metrics": {
            "replay_macro_f1": metrics[replay_variant]["macro_f1"],
            "candidate_macro_f1": metrics[candidate_variant]["macro_f1"],
            "replay_schema_valid_rate": metrics[replay_variant]["schema_valid_rate"],
            "candidate_schema_valid_rate": metrics[candidate_variant][
                "schema_valid_rate"
            ],
            "nontruncated_raw_response_exact_rate": report[
                "nontruncated_exact_replay"
            ]["rate"],
            "original_truncations_rescued": report[
                "original_truncation_recovery"
            ]["rescued"],
            "original_truncation_count": report[
                "original_truncation_recovery"
            ]["original_count"],
            "candidate_minus_replay_macro_f1_ci95": report[
                "paired_candidate_minus_replay"
            ]["bootstrap_95_ci"],
        },
        "evidence": {
            "gate_report_sha256": report_sha256,
            "run_manifest_sha256": manifest_sha256,
            "input_hashes": report.get("input_hashes", {}),
        },
        "feedback_contract": {
            "return_files": [
                "new_plan/logs/p1_structured_decoding_offline_token_budget.log",
                "diagnostics/offline_structured_decoding_feedback.json",
                "diagnostics/offline_structured_decoding_token_budget_gate.json",
                "run_manifest.json",
            ],
            "do_not": [
                "change_preregistered_thresholds",
                "inspect_human_gold_for_candidate_selection",
                "start_sft_dpo_or_grpo",
            ],
        },
    }


def render_markdown(packet: Mapping[str, Any]) -> str:
    failed_replay = packet["failed_replay_checks"] or ["none"]
    failed_candidate = packet["failed_candidate_checks"] or ["none"]
    metrics = packet["headline_metrics"]
    return "\n".join(
        [
            "# Offline Structured Decoding Feedback",
            "",
            f"- Decision: `{packet['decision']}`",
            f"- Next work package: `{packet['next_work_package']}`",
            f"- Authorization: `{packet['authorization']}`",
            f"- Failed replay checks: {', '.join(failed_replay)}",
            f"- Failed candidate checks: {', '.join(failed_candidate)}",
            (
                "- Replay/candidate Macro-F1: "
                f"{metrics['replay_macro_f1']:.6f} / "
                f"{metrics['candidate_macro_f1']:.6f}"
            ),
            (
                "- Nontruncated raw-response exact rate: "
                f"{metrics['nontruncated_raw_response_exact_rate']:.6f}"
            ),
            (
                "- Original truncations rescued: "
                f"{metrics['original_truncations_rescued']} / "
                f"{metrics['original_truncation_count']}"
            ),
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path(
            "reports/baselines/qwen2_5_7b_dev/v7_offline_deterministic_768"
        ),
    )
    args = parser.parse_args()

    diagnostics_dir = args.experiment_dir / "diagnostics"
    report_path = diagnostics_dir / "offline_structured_decoding_token_budget_gate.json"
    manifest_path = args.experiment_dir / "run_manifest.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    packet = build_feedback_packet(
        report,
        report_sha256=file_sha256(report_path),
        manifest_sha256=file_sha256(manifest_path),
    )
    packet_path = diagnostics_dir / "offline_structured_decoding_feedback.json"
    markdown_path = diagnostics_dir / "offline_structured_decoding_feedback.md"
    write_json_atomic(packet_path, packet)
    write_text_atomic(markdown_path, render_markdown(packet))
    print(
        json.dumps(
            {
                "event": "p1_structured_decoding_feedback_packet_complete",
                "status": "PASS",
                "decision": packet["decision"],
                "next_work_package": packet["next_work_package"],
                "packet_file": str(packet_path),
                "packet_sha256": file_sha256(packet_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
