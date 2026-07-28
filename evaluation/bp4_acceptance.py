"""Generic paired acceptance gate for BP4 post-training candidates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from evaluation.keyword_evaluator import file_sha256, write_json_atomic


REQUIRED_METRICS = (
    "micro_f1",
    "macro_f1",
    "schema_valid_rate",
    "hallucination_keyword_rate",
    "mean_output_tokens",
    "max_token_output_count",
)


def validate_evaluation_packet(packet: Mapping[str, Any]) -> None:
    if packet.get("packet_version") != "bp4-paired-evaluation-v1":
        raise ValueError("Unsupported BP4 evaluation packet")
    if packet.get("selection_split") != "teacher_dev":
        raise ValueError("Acceptance must use teacher_dev")
    if packet.get("human_gold_used") is not False:
        raise ValueError("Human gold must not participate in BP4 stage acceptance")
    if packet.get("sample_count") != 331:
        raise ValueError("BP4 dev coverage must be exactly 331")
    if packet.get("target_contract_version") != "keyword-json-target-v2":
        raise ValueError("BP4 acceptance requires keyword-json-target-v2")
    if packet.get("bootstrap_iterations") != 10000:
        raise ValueError("BP4 acceptance requires 10000 bootstrap iterations")
    if packet.get("bootstrap_seed") != 42:
        raise ValueError("BP4 acceptance requires bootstrap seed 42")
    for name in (
        "selection_data_sha256",
        "baseline_predictions_sha256",
        "candidate_predictions_sha256",
    ):
        if not isinstance(packet.get(name), str) or len(packet[name]) != 64:
            raise ValueError(f"Missing traceable packet field: {name}")
    for name in (
        "baseline_model_id",
        "candidate_model_id",
        "baseline_checkpoint",
        "candidate_checkpoint",
        "baseline_stage",
    ):
        if not isinstance(packet.get(name), str) or not packet[name]:
            raise ValueError(f"Missing model-lineage packet field: {name}")
    for side in ("baseline", "candidate"):
        metrics = packet.get(side)
        if not isinstance(metrics, dict):
            raise ValueError(f"Missing {side} metrics")
        for name in REQUIRED_METRICS:
            if not isinstance(metrics.get(name), (int, float)):
                raise ValueError(f"Missing numeric {side}.{name}")
    paired = packet.get("paired")
    if not isinstance(paired, dict):
        raise ValueError("Missing paired metrics")
    ci = paired.get("macro_f1_bootstrap_95_ci")
    if (
        not isinstance(ci, list)
        or len(ci) != 2
        or not all(isinstance(value, (int, float)) for value in ci)
        or ci[0] > ci[1]
    ):
        raise ValueError("Invalid paired bootstrap CI")
    delta = paired.get("macro_f1_delta_candidate_minus_baseline")
    expected_delta = (
        packet["candidate"]["macro_f1"] - packet["baseline"]["macro_f1"]
    )
    if not isinstance(delta, (int, float)) or abs(delta - expected_delta) > 1e-12:
        raise ValueError("Paired macro-F1 delta is inconsistent with metrics")


def decide_stage(
    packet: Mapping[str, Any],
    gates: Mapping[str, Any],
    *,
    stage: str,
    candidate_checkpoint: str,
) -> Dict[str, Any]:
    if stage not in {"sft", "dpo", "grpo"}:
        raise ValueError(f"Unsupported BP4 stage: {stage}")
    validate_evaluation_packet(packet)
    allowed_baseline_stages = {
        "sft": {"base"},
        "dpo": {"sft"},
        "grpo": {"sft", "dpo"},
    }
    if packet["baseline_stage"] not in allowed_baseline_stages[stage]:
        raise ValueError(f"Invalid direct upstream for {stage}: {packet['baseline_stage']}")
    if packet["candidate_checkpoint"] != candidate_checkpoint:
        raise ValueError("Candidate checkpoint does not match evaluation packet")
    baseline = packet["baseline"]
    candidate = packet["candidate"]
    paired = packet["paired"]
    output_ratio = candidate["mean_output_tokens"] / max(
        baseline["mean_output_tokens"], 1e-12
    )
    checks = {
        "complete_dev_coverage": (
            packet["sample_count"] == 331
            if gates["require_complete_dev_coverage"]
            else True
        ),
        "paired_macro_f1_delta_min": (
            paired["macro_f1_delta_candidate_minus_baseline"]
            >= gates["paired_macro_f1_delta_min"]
        ),
        "paired_macro_f1_ci_low_gt": (
            paired["macro_f1_bootstrap_95_ci"][0]
            > gates["paired_macro_f1_ci_low_gt"]
        ),
        "micro_f1_delta_min": (
            candidate["micro_f1"] - baseline["micro_f1"]
            >= gates["micro_f1_delta_min"]
        ),
        "schema_valid_rate_min": (
            candidate["schema_valid_rate"] >= gates["schema_valid_rate_min"]
        ),
        "schema_valid_rate_delta_min": (
            candidate["schema_valid_rate"] - baseline["schema_valid_rate"]
            >= gates["schema_valid_rate_delta_min"]
        ),
        "hallucination_keyword_rate_max": (
            candidate["hallucination_keyword_rate"]
            <= gates["hallucination_keyword_rate_max"]
        ),
        "hallucination_keyword_rate_delta_max": (
            candidate["hallucination_keyword_rate"]
            - baseline["hallucination_keyword_rate"]
            <= gates["hallucination_keyword_rate_delta_max"]
        ),
        "mean_output_tokens_ratio_max": (
            output_ratio <= gates["mean_output_tokens_ratio_max"]
        ),
        "max_token_output_count_delta_max": (
            candidate["max_token_output_count"]
            - baseline["max_token_output_count"]
            <= gates["max_token_output_count_delta_max"]
        ),
        "human_gold_not_used": packet["human_gold_used"] is False,
        "challenge_not_used_for_selection": (
            packet.get("challenge_usage") == "diagnostic_only"
            if gates["challenge_is_diagnostic_only"]
            else True
        ),
    }
    accepted = all(checks.values())
    return {
        "report_version": "bp4-stage-acceptance-v1",
        "stage": stage,
        "status": "PASS",
        "decision": f"ACCEPT_{stage.upper()}_CANDIDATE" if accepted else f"REJECT_{stage.upper()}_CANDIDATE",
        "candidate_checkpoint": candidate_checkpoint,
        "baseline_checkpoint": packet["baseline_checkpoint"],
        "baseline_model_id": packet.get("baseline_model_id"),
        "candidate_model_id": packet.get("candidate_model_id"),
        "selection_split": "teacher_dev",
        "challenge_usage": "diagnostic_only",
        "human_gold_used": False,
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "paired": paired,
        "baseline": baseline,
        "candidate": candidate,
        "mean_output_tokens_ratio": output_ratio,
    }


def authorization_candidate(
    acceptance: Mapping[str, Any],
    *,
    next_stage: str,
    bp2_config_sha256: str,
    model_path: str,
    dataset_sha256: str | None = None,
    val_dataset_sha256: str | None = None,
) -> Dict[str, Any]:
    if acceptance.get("report_version") != "bp4-stage-acceptance-v1":
        raise ValueError("Unsupported upstream acceptance report")
    if acceptance.get("status") != "PASS":
        raise ValueError("Upstream acceptance report is not valid")
    expected_decision = f"ACCEPT_{acceptance.get('stage', '').upper()}_CANDIDATE"
    if acceptance.get("decision") != expected_decision:
        raise ValueError("Cannot authorize from rejected upstream candidate")
    checks = acceptance.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError("Upstream acceptance checks are not all passing")
    if next_stage not in {"dpo", "grpo"}:
        raise ValueError("Only downstream DPO/GRPO authorization is supported")
    payload = {
        "authorization_version": "bp4-stage-authorization-candidate-v1",
        "activation_status": "CANDIDATE_NOT_ACTIVE",
        "stage": next_stage,
        "decision": f"AUTHORIZE_{next_stage.upper()}_TRAINING",
        "config_sha256": bp2_config_sha256,
        "model_path": model_path,
        "upstream_acceptance_decision": acceptance["decision"],
        "upstream_candidate_checkpoint": acceptance["candidate_checkpoint"],
        "human_approval_required": True,
    }
    if dataset_sha256 is not None:
        payload["dataset_sha256"] = dataset_sha256
    if val_dataset_sha256 is not None:
        payload["val_dataset_sha256"] = val_dataset_sha256
    return payload


def write_acceptance(
    packet_file: Path,
    output_file: Path,
    gates: Mapping[str, Any],
    *,
    stage: str,
    candidate_checkpoint: str,
) -> Dict[str, Any]:
    packet = json.loads(packet_file.read_text(encoding="utf-8"))
    report = decide_stage(
        packet, gates, stage=stage, candidate_checkpoint=candidate_checkpoint
    )
    report["input_packet_sha256"] = file_sha256(packet_file)
    write_json_atomic(output_file, report)
    return report
