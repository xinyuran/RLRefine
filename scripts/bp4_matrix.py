"""Prepare BP4 matrix engineering and stage-authorization candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from evaluation.bp4_acceptance import (
    authorization_candidate,
    write_acceptance,
)
from evaluation.keyword_evaluator import file_sha256, write_json_atomic
from scripts.prepare_experiment_manifest import (
    _environment_lineage,
    _git_lineage,
    _source_lineage,
)


SCHEMA_VERSION = "structalign-bp4-controlled-matrix-v1"
REPORT_VERSION = "bp4-controlled-model-matrix-v1"


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True))


def load_config(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Invalid BP4 config")
    return value


def _read_gate(path: Path, expected_sha256: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"Gate hash mismatch for {path}: {actual}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or not str(
        payload.get("decision", "ACCEPT")
    ).startswith("ACCEPT"):
        raise ValueError(f"Dependency gate not accepted: {path}")
    return payload


def validate_dependencies(
    root: Path,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    checked: Dict[str, Any] = {}
    for name in ("bp2_gate", "bp3_gate", "preference_gate", "reward_gate"):
        spec = config["pipeline"][name]
        path = root / spec["path"]
        _read_gate(path, spec["sha256"])
        checked[name] = {"path": spec["path"], "sha256": spec["sha256"]}
    for name in ("selection", "diagnostic"):
        spec = config["evaluation"][name]
        path = root / spec["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        if actual != spec["sha256"]:
            raise ValueError(f"{name} data hash mismatch")
        rows = sum(
            1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        if rows != int(spec["rows"]):
            raise ValueError(f"{name} row count mismatch")
        checked[name] = {"path": spec["path"], "sha256": actual, "rows": rows}
    if config["evaluation"]["forbidden_selection"] != "frozen_human_gold":
        raise ValueError("BP4 must forbid human gold for stage selection")
    return checked


def _bp2_config(root: Path, config: Mapping[str, Any]) -> tuple[Dict[str, Any], Path]:
    path = root / config["pipeline"]["config"]
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "structalign-bp2-pipeline-v1":
        raise ValueError("BP4 requires the accepted BP2 pipeline config")
    return value, path


def base_task(config: Mapping[str, Any], bp2: Mapping[str, Any]) -> Dict[str, Any]:
    evaluation = config["evaluation"]
    return {
        "task_version": "bp4-base-v2-evaluation-task-v1",
        "matrix_id": "B1_V2",
        "stage": "base",
        "execution": "inference_only_no_training",
        "model_path": bp2["model"]["base_path"],
        "target_contract_version": evaluation["target_contract_version"],
        "selection": dict(evaluation["selection"]),
        "diagnostic": dict(evaluation["diagnostic"]),
        "forbidden_for_selection": evaluation["forbidden_selection"],
        "decoding": dict(evaluation["decoding"]),
        "required_outputs": {
            "dev_predictions": "reports/bp4_controlled_matrix/base_v2/dev_predictions.jsonl",
            "dev_evaluation": "reports/bp4_controlled_matrix/base_v2/dev_evaluation.json",
            "challenge_predictions": "reports/bp4_controlled_matrix/base_v2/challenge_predictions.jsonl",
            "challenge_evaluation": "reports/bp4_controlled_matrix/base_v2/challenge_evaluation.json",
            "base_packet": "reports/bp4_controlled_matrix/base_v2/base_evaluation_packet.json",
        },
        "completion_gate": {
            "dev_rows": 331,
            "challenge_rows": 140,
            "dev_coverage_exact": True,
            "challenge_coverage_exact": True,
            "human_gold_used": False,
        },
        "base_packet_contract": {
            "packet_version": "bp4-base-evaluation-v1",
            "status": "PASS",
            "target_contract_version": evaluation["target_contract_version"],
            "selection_data_sha256": evaluation["selection"]["sha256"],
            "diagnostic_data_sha256": evaluation["diagnostic"]["sha256"],
            "dev_predictions_sha256": "required",
            "challenge_predictions_sha256": "required",
            "decoding": dict(evaluation["decoding"]),
            "challenge_usage": "diagnostic_only",
            "human_gold_used": False,
        },
    }


def matrix_plan(config: Mapping[str, Any]) -> Dict[str, Any]:
    statuses = {
        "B1_V2": {
            "status": "READY",
            "blockers": [],
            "authorization_required": False,
        },
        "E1_SFT": {
            "status": "BLOCKED",
            "blockers": [
                "missing_base_v2_evaluation_packet",
                "missing_active_stage_authorization:reports/authorizations/bp2_sft.json",
            ],
            "authorization_required": True,
        },
        "E2_DPO": {
            "status": "BLOCKED",
            "blockers": [
                "missing_sft_acceptance:reports/model_gates/bp2_sft_acceptance.json",
                "missing_active_stage_authorization:reports/authorizations/bp2_dpo.json",
            ],
            "authorization_required": True,
        },
        "E3_GRPO_FROM_SFT": {
            "status": "BLOCKED",
            "blockers": [
                "missing_sft_upstream_acceptance:reports/model_gates/bp2_upstream_acceptance.json",
                "missing_active_stage_authorization:reports/authorizations/bp2_grpo.json",
            ],
            "authorization_required": True,
        },
        "E4_GRPO_FROM_DPO": {
            "status": "BLOCKED",
            "blockers": [
                "missing_dpo_upstream_acceptance:reports/model_gates/bp2_upstream_acceptance.json",
                "missing_active_stage_authorization:reports/authorizations/bp2_grpo.json",
            ],
            "authorization_required": True,
        },
    }
    return {
        "plan_version": REPORT_VERSION,
        "policy": config["policy"],
        "rows": [
            {**dict(row), **statuses[row["id"]]} for row in config["matrix"]
        ],
        "selection_policy": (
            "teacher_dev_only_challenge_diagnostic_human_gold_forbidden"
        ),
        "training_started": False,
    }


def prepare_package(
    root: Path,
    config_path: Path,
) -> Dict[str, Any]:
    config = load_config(config_path)
    dependencies = validate_dependencies(root, config)
    bp2, bp2_path = _bp2_config(root, config)
    output = config["outputs"]
    report_dir = root / output["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    plan = matrix_plan(config)
    task = base_task(config, bp2)
    request = {
        "authorization_version": "bp4-sft-authorization-request-v1",
        "activation_status": "BLOCKED_MISSING_BASE_V2_EVALUATION",
        "stage": "sft",
        "proposed_decision": "AUTHORIZE_SFT_TRAINING",
        "active_path": bp2["stages"]["sft"]["authorization_file"],
        "config_sha256": file_sha256(bp2_path),
        "model_path": bp2["model"]["base_path"],
        "prerequisites": [
            "base_v2_evaluation_packet_status_PASS",
            "dev_coverage_331",
            "challenge_coverage_140",
            "human_gold_used_false",
            "explicit_human_approval",
        ],
    }
    write_json_atomic(root / output["matrix_plan"], plan)
    write_json_atomic(root / output["base_task"], task)
    write_json_atomic(root / output["sft_authorization_candidate"], request)

    active_auth_paths = [
        root / bp2["stages"][stage]["authorization_file"]
        for stage in ("sft", "dpo", "grpo")
    ]
    checks = {
        "accepted_bp2_gate_locked": True,
        "accepted_bp3_gate_locked": True,
        "preference_gate_locked": True,
        "reward_gate_locked": True,
        "base_v2_is_first_and_ready": plan["rows"][0]["id"] == "B1_V2"
        and plan["rows"][0]["status"] == "READY",
        "sft_blocked_before_base_and_authorization": (
            plan["rows"][1]["status"] == "BLOCKED"
        ),
        "dpo_requires_sft_acceptance": (
            "missing_sft_acceptance:reports/model_gates/bp2_sft_acceptance.json"
            in plan["rows"][2]["blockers"]
        ),
        "grpo_requires_upstream_acceptance": all(
            any("upstream_acceptance" in blocker for blocker in row["blockers"])
            for row in plan["rows"][3:]
        ),
        "human_gold_forbidden_for_selection": (
            task["forbidden_for_selection"] == "frozen_human_gold"
        ),
        "challenge_is_diagnostic_only": config["acceptance"][
            "challenge_is_diagnostic_only"
        ],
        "no_active_stage_authorizations_created": not any(
            path.exists() for path in active_auth_paths
        ),
        "training_not_started": True,
    }
    manifest = {
        "manifest_version": REPORT_VERSION,
        "config": {
            "path": str(config_path.relative_to(root)).replace("\\", "/"),
            "sha256": file_sha256(config_path),
        },
        "bp2_config": {
            "path": config["pipeline"]["config"],
            "sha256": file_sha256(bp2_path),
        },
        "dependencies": dependencies,
        "outputs": {
            name: {
                "path": output[name],
                "sha256": file_sha256(root / output[name]),
            }
            for name in (
                "matrix_plan",
                "base_task",
                "sft_authorization_candidate",
            )
        },
        "checks": checks,
        "source": _source_lineage(root),
        "git": _git_lineage(root),
        "environment": _environment_lineage(),
        "training_started": False,
    }
    write_json_atomic(root / output["manifest"], manifest)
    gate = {
        "report_version": REPORT_VERSION,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "decision": (
            "ACCEPT_BP4_MATRIX_ENGINEERING"
            if all(checks.values())
            else "REJECT_BP4_MATRIX_ENGINEERING"
        ),
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "manifest_sha256": file_sha256(root / output["manifest"]),
        "next_execution": "BP4_BASE_V2_EVALUATION",
        "authorization": "base_inference_only; sft_training_still_blocked",
        "training_started": False,
    }
    write_json_atomic(root / output["gate"], gate)
    feedback = {
        "packet_version": "bp4-feedback-v1",
        "decision": gate["decision"],
        "failed_checks": gate["failed_checks"],
        "manifest_sha256": gate["manifest_sha256"],
        "gate_sha256": file_sha256(root / output["gate"]),
        "next_execution": gate["next_execution"],
        "authorization": gate["authorization"],
        "training_started": False,
    }
    write_json_atomic(root / output["feedback"], feedback)
    emit(
        "bp4_gate_complete",
        status=gate["status"],
        decision=gate["decision"],
        manifest_file=output["manifest"],
        manifest_sha256=gate["manifest_sha256"],
        gate_file=output["gate"],
        gate_sha256=file_sha256(root / output["gate"]),
        feedback_file=output["feedback"],
        feedback_sha256=file_sha256(root / output["feedback"]),
        training_started=False,
    )
    if gate["status"] != "PASS":
        raise ValueError(f"BP4 gate failed: {gate['failed_checks']}")
    return gate


def build_sft_authorization_candidate(
    root: Path,
    config_path: Path,
    base_packet_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    config = load_config(config_path)
    validate_dependencies(root, config)
    bp2, bp2_path = _bp2_config(root, config)
    packet = json.loads(base_packet_path.read_text(encoding="utf-8"))
    required = {
        "packet_version": "bp4-base-evaluation-v1",
        "status": "PASS",
        "target_contract_version": config["evaluation"]["target_contract_version"],
        "dev_rows": 331,
        "challenge_rows": 140,
        "human_gold_used": False,
        "challenge_usage": "diagnostic_only",
        "selection_data_sha256": config["evaluation"]["selection"]["sha256"],
        "diagnostic_data_sha256": config["evaluation"]["diagnostic"]["sha256"],
        "decoding": config["evaluation"]["decoding"],
    }
    for name, expected in required.items():
        if packet.get(name) != expected:
            raise ValueError(f"Base packet prerequisite failed: {name}")
    for name in ("dev_predictions_sha256", "challenge_predictions_sha256"):
        if not isinstance(packet.get(name), str) or len(packet[name]) != 64:
            raise ValueError(f"Base packet prerequisite failed: {name}")
    active_path = root / bp2["stages"]["sft"]["authorization_file"]
    if output_path.resolve() == active_path.resolve():
        raise ValueError("Authorization candidates cannot be written to the active path")
    candidate = {
        "authorization_version": "bp4-stage-authorization-candidate-v1",
        "activation_status": "CANDIDATE_NOT_ACTIVE",
        "stage": "sft",
        "decision": "AUTHORIZE_SFT_TRAINING",
        "config_sha256": file_sha256(bp2_path),
        "model_path": bp2["model"]["base_path"],
        "dataset_sha256": file_sha256(
            root / bp2["stages"]["sft"]["dataset"]
        ),
        "val_dataset_sha256": file_sha256(
            root / bp2["stages"]["sft"]["val_dataset"]
        ),
        "base_packet_path": str(base_packet_path),
        "base_packet_sha256": file_sha256(base_packet_path),
        "human_approval_required": True,
        "active_path": bp2["stages"]["sft"]["authorization_file"],
    }
    write_json_atomic(output_path, candidate)
    return candidate


def build_downstream_candidate(
    root: Path,
    config_path: Path,
    acceptance_path: Path,
    *,
    next_stage: str,
    output_path: Path,
) -> Dict[str, Any]:
    config = load_config(config_path)
    validate_dependencies(root, config)
    bp2, bp2_path = _bp2_config(root, config)
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    stage_config = bp2["stages"][next_stage]
    active_path = root / stage_config["authorization_file"]
    if output_path.resolve() == active_path.resolve():
        raise ValueError("Authorization candidates cannot be written to the active path")
    candidate = authorization_candidate(
        acceptance,
        next_stage=next_stage,
        bp2_config_sha256=file_sha256(bp2_path),
        model_path=acceptance["candidate_checkpoint"],
        dataset_sha256=file_sha256(root / stage_config["dataset"]),
        val_dataset_sha256=file_sha256(root / stage_config["val_dataset"]),
    )
    candidate["upstream_acceptance_path"] = str(acceptance_path)
    candidate["upstream_acceptance_sha256"] = file_sha256(acceptance_path)
    candidate["active_path"] = stage_config["authorization_file"]
    write_json_atomic(output_path, candidate)
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/bp4_controlled_model_matrix_v1.yaml"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sft = sub.add_parser("sft-authorization-candidate")
    sft.add_argument("--base-packet", type=Path, required=True)
    sft.add_argument("--output", type=Path, required=True)
    decide = sub.add_parser("decide-stage")
    decide.add_argument("--stage", choices=("sft", "dpo", "grpo"), required=True)
    decide.add_argument("--packet", type=Path, required=True)
    decide.add_argument("--candidate-checkpoint", required=True)
    decide.add_argument("--output", type=Path, required=True)
    downstream = sub.add_parser("downstream-authorization-candidate")
    downstream.add_argument("--acceptance", type=Path, required=True)
    downstream.add_argument("--next-stage", choices=("dpo", "grpo"), required=True)
    downstream.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_config(config_path)
    resolve_from_root = lambda path: path if path.is_absolute() else root / path
    try:
        if args.command == "prepare":
            prepare_package(root, config_path)
        elif args.command == "sft-authorization-candidate":
            build_sft_authorization_candidate(
                root,
                config_path,
                resolve_from_root(args.base_packet),
                resolve_from_root(args.output),
            )
            emit("bp4_sft_authorization_candidate", status="PASS", output=str(args.output))
        elif args.command == "decide-stage":
            report = write_acceptance(
                resolve_from_root(args.packet),
                resolve_from_root(args.output),
                config["acceptance"],
                stage=args.stage,
                candidate_checkpoint=args.candidate_checkpoint,
            )
            emit(
                "bp4_stage_acceptance",
                status="PASS",
                decision=report["decision"],
                output=str(args.output),
            )
        else:
            build_downstream_candidate(
                root,
                config_path,
                resolve_from_root(args.acceptance),
                next_stage=args.next_stage,
                output_path=resolve_from_root(args.output),
            )
            emit(
                "bp4_downstream_authorization_candidate",
                status="PASS",
                next_stage=args.next_stage,
                output=str(args.output),
            )
        return 0
    except Exception as exc:
        emit(
            "bp4_command",
            status="FAIL",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
