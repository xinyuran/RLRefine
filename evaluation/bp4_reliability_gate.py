"""Prospective two-level BP4 gate amendment without rewriting legacy evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from evaluation.bp4_acceptance import decide_stage, validate_evaluation_packet
from evaluation.keyword_evaluator import file_sha256, write_json_atomic


SCHEMA_VERSION = "structalign-bp4-reliability-repair-v1"


def load_config(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Invalid BP4 reliability-repair config")
    return value


def _research_checks(
    packet: Mapping[str, Any], gates: Mapping[str, Any]
) -> Dict[str, bool]:
    baseline = packet["baseline"]
    candidate = packet["candidate"]
    paired = packet["paired"]
    return {
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
        "schema_valid_rate_delta_min": (
            candidate["schema_valid_rate"] - baseline["schema_valid_rate"]
            >= gates["schema_valid_rate_delta_min"]
        ),
        "hallucination_keyword_rate_delta_max": (
            candidate["hallucination_keyword_rate"]
            - baseline["hallucination_keyword_rate"]
            <= gates["hallucination_keyword_rate_delta_max"]
        ),
        "human_gold_not_used": packet["human_gold_used"] is False,
        "challenge_not_used_for_selection": (
            packet.get("challenge_usage") == "diagnostic_only"
            if gates["challenge_is_diagnostic_only"]
            else True
        ),
    }


def build_amendment(
    packet: Mapping[str, Any],
    legacy_acceptance: Mapping[str, Any],
    legacy_gate: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    packet_sha256: str,
    legacy_acceptance_sha256: str,
    legacy_gate_sha256: str,
) -> Dict[str, Any]:
    validate_evaluation_packet(packet)
    if legacy_acceptance.get("decision") != "REJECT_SFT_CANDIDATE":
        raise ValueError("Amendment requires the preserved legacy SFT rejection")
    if legacy_gate.get("decision") != "REJECT_SFT_CANDIDATE":
        raise ValueError("Legacy package gate decision is not the expected rejection")
    if legacy_gate.get("acceptance_sha256") != legacy_acceptance_sha256:
        raise ValueError("Legacy gate does not reference the supplied acceptance")
    if legacy_gate.get("paired_packet_sha256") != packet_sha256:
        raise ValueError("Legacy gate does not reference the supplied paired packet")

    research_checks = _research_checks(
        packet, config["research_progression_gate"]
    )
    deployment = decide_stage(
        packet,
        config["deployment_gate"],
        stage="sft",
        candidate_checkpoint=packet["candidate_checkpoint"],
    )
    research_passed = all(research_checks.values())
    deployment_passed = deployment["decision"] == "ACCEPT_SFT_CANDIDATE"
    return {
        "report_version": "bp4-two-level-gate-amendment-v1",
        "status": "PASS",
        "amendment_policy": "prospective_only_legacy_decision_preserved",
        "legacy_decision": "REJECT_SFT_CANDIDATE",
        "legacy_acceptance_sha256": legacy_acceptance_sha256,
        "legacy_gate_sha256": legacy_gate_sha256,
        "paired_packet_sha256": packet_sha256,
        "stage": "sft",
        "candidate_checkpoint": packet["candidate_checkpoint"],
        "research_progression_decision": (
            "ACCEPT_SFT_RESEARCH_PROGRESSION"
            if research_passed
            else "REJECT_SFT_RESEARCH_PROGRESSION"
        ),
        "research_progression_checks": research_checks,
        "research_progression_failed_checks": [
            name for name, passed in research_checks.items() if not passed
        ],
        "deployment_decision": (
            "ACCEPT_SFT_DEPLOYMENT"
            if deployment_passed
            else "REJECT_SFT_DEPLOYMENT"
        ),
        "deployment_checks": deployment["checks"],
        "deployment_failed_checks": deployment["failed_checks"],
        "downstream_training_authorized": False,
        "human_approval_required": research_passed,
        "next_experiment_candidate": (
            "E2_DPO_RELIABILITY" if research_passed else None
        ),
    }


def authorization_candidate(
    amendment: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    if (
        amendment.get("report_version") != "bp4-two-level-gate-amendment-v1"
        or amendment.get("status") != "PASS"
        or amendment.get("research_progression_decision")
        != "ACCEPT_SFT_RESEARCH_PROGRESSION"
    ):
        raise ValueError("SFT research-progression gate is not accepted")
    checks = amendment.get("research_progression_checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError("SFT research-progression checks are incomplete")
    prereg = config["dpo_repair_preregistration"]
    return {
        "authorization_version": "bp4-reliability-repair-candidate-v1",
        "activation_status": "CANDIDATE_NOT_ACTIVE",
        "stage": "dpo",
        "decision": "AUTHORIZE_DPO_TRAINING",
        "approved_scope": "BP4_DPO_RELIABILITY_REPAIR_ONLY",
        "matrix_id": prereg["matrix_id"],
        "model_path": amendment["candidate_checkpoint"],
        "upstream_research_decision": amendment[
            "research_progression_decision"
        ],
        "deployment_status": amendment["deployment_decision"],
        "optimization_objectives": [
            "schema_validity",
            "faithfulness",
            "output_length",
            "max_token_truncation",
        ],
        "checkpoint_selection": prereg["checkpoint_selection"],
        "human_approval_required": True,
        "downstream_grpo_authorized": False,
    }


def prepare(
    config_path: Path,
    packet_path: Path,
    legacy_acceptance_path: Path,
    legacy_gate_path: Path,
    root: Path,
) -> Dict[str, Any]:
    config = load_config(config_path)
    packet_hash = file_sha256(packet_path)
    acceptance_hash = file_sha256(legacy_acceptance_path)
    expected = config["inputs"]["expected"]
    if packet_hash != expected["paired_packet_sha256"]:
        raise ValueError("Paired packet hash mismatch")
    if acceptance_hash != expected["legacy_acceptance_sha256"]:
        raise ValueError("Legacy acceptance hash mismatch")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    legacy_acceptance = json.loads(
        legacy_acceptance_path.read_text(encoding="utf-8")
    )
    legacy_gate = json.loads(legacy_gate_path.read_text(encoding="utf-8"))
    gate_hash = file_sha256(legacy_gate_path)
    amendment = build_amendment(
        packet,
        legacy_acceptance,
        legacy_gate,
        config,
        packet_sha256=packet_hash,
        legacy_acceptance_sha256=acceptance_hash,
        legacy_gate_sha256=gate_hash,
    )
    candidate = authorization_candidate(amendment, config)
    outputs = config["outputs"]
    amendment_path = root / outputs["amendment"]
    candidate_path = root / outputs["dpo_authorization_candidate"]
    write_json_atomic(amendment_path, amendment)
    candidate["upstream_amendment_path"] = outputs["amendment"]
    candidate["upstream_amendment_sha256"] = file_sha256(amendment_path)
    write_json_atomic(candidate_path, candidate)

    checks = {
        "legacy_decision_preserved": (
            amendment["legacy_decision"] == "REJECT_SFT_CANDIDATE"
        ),
        "research_progression_gate_passed": (
            amendment["research_progression_decision"]
            == "ACCEPT_SFT_RESEARCH_PROGRESSION"
        ),
        "deployment_gate_still_failed": (
            amendment["deployment_decision"] == "REJECT_SFT_DEPLOYMENT"
        ),
        "dpo_candidate_is_inactive": (
            candidate["activation_status"] == "CANDIDATE_NOT_ACTIVE"
        ),
        "grpo_not_authorized": candidate["downstream_grpo_authorized"] is False,
        "human_gold_not_used": packet["human_gold_used"] is False,
        "challenge_diagnostic_only": (
            packet["challenge_usage"] == "diagnostic_only"
        ),
        "training_not_started": True,
    }
    manifest = {
        "manifest_version": "bp4-reliability-repair-v1",
        "config_sha256": file_sha256(config_path),
        "inputs": {
            "paired_packet_sha256": packet_hash,
            "legacy_acceptance_sha256": acceptance_hash,
            "legacy_gate_sha256": gate_hash,
        },
        "outputs": {
            "amendment_sha256": file_sha256(amendment_path),
            "dpo_authorization_candidate_sha256": file_sha256(candidate_path),
        },
        "checks": checks,
        "training_started": False,
    }
    manifest_path = root / outputs["manifest"]
    write_json_atomic(manifest_path, manifest)
    accepted = all(checks.values())
    gate = {
        "report_version": "bp4-reliability-repair-v1",
        "status": "PASS" if accepted else "FAIL",
        "decision": (
            "ACCEPT_BP4_TWO_LEVEL_GATE_AMENDMENT"
            if accepted
            else "REJECT_BP4_TWO_LEVEL_GATE_AMENDMENT"
        ),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "manifest_sha256": file_sha256(manifest_path),
        "next_execution": "HUMAN_REVIEW_DPO_RELIABILITY_AUTHORIZATION",
        "training_started": False,
    }
    gate_path = root / outputs["gate"]
    write_json_atomic(gate_path, gate)
    feedback = {
        "packet_version": "bp4-reliability-repair-feedback-v1",
        "decision": gate["decision"],
        "gate_sha256": file_sha256(gate_path),
        "manifest_sha256": gate["manifest_sha256"],
        "next_execution": gate["next_execution"],
        "dpo_training_authorized": False,
        "grpo_training_authorized": False,
        "training_started": False,
    }
    write_json_atomic(root / outputs["feedback"], feedback)
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/bp4_reliability_repair_v1.yaml"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--paired-packet", type=Path)
    parser.add_argument("--legacy-acceptance", type=Path)
    parser.add_argument("--legacy-gate", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_config(config_path)

    def input_path(cli_value: Path | None, key: str) -> Path:
        value = cli_value or Path(config["inputs"][key])
        return value if value.is_absolute() else root / value

    try:
        gate = prepare(
            config_path,
            input_path(args.paired_packet, "paired_packet"),
            input_path(args.legacy_acceptance, "legacy_acceptance"),
            input_path(args.legacy_gate, "legacy_gate"),
            root,
        )
        print(json.dumps({
            "event": "bp4_reliability_gate_complete",
            "status": gate["status"],
            "decision": gate["decision"],
            "next_execution": gate["next_execution"],
            "training_started": False,
        }, ensure_ascii=False, sort_keys=True))
        return 0 if gate["status"] == "PASS" else 1
    except Exception as exc:
        print(json.dumps({
            "event": "bp4_reliability_gate_complete",
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "training_started": False,
        }, ensure_ascii=False, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
