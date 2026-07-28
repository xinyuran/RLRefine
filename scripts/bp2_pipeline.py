"""Config-driven, gated SFT/DPO/GRPO engineering pipeline for BP2."""
import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import yaml

from evaluation.keyword_evaluator import file_sha256
from scripts.prepare_experiment_manifest import (
    _environment_lineage,
    _git_lineage,
    _source_lineage,
)


SCHEMA_VERSION = "structalign-bp2-pipeline-v1"
PIPELINE_VERSION = "bp2-post-training-engineering-v1"
STAGES = ("sft", "dpo", "grpo")
ENTRYPOINTS = {"sft": {"sft"}, "dpo": {"dpo", "rlhf"}, "grpo": {"rlhf"}}
AUTH_DECISIONS = {
    "sft": "AUTHORIZE_SFT_TRAINING",
    "dpo": "AUTHORIZE_DPO_TRAINING",
    "grpo": "AUTHORIZE_GRPO_TRAINING",
}


def emit(event: str, **payload: Any) -> None:
    print(
        json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


def load_config(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("BP2 config must be a mapping")
    return config


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config field {key!r} must be a mapping")
    return value


def _jsonl_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def validate_config(
    config: Mapping[str, Any],
    repo_root: Path,
    *,
    require_bp1_assets: bool,
) -> Dict[str, Any]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    pipeline = _mapping(config, "pipeline")
    if pipeline.get("dataset_version") != "keyword-v2.0.0":
        raise ValueError("BP2 must consume keyword-v2.0.0")
    if pipeline.get("target_contract_version") != "keyword-json-target-v2":
        raise ValueError("BP2 target contract drift")
    model = _mapping(config, "model")
    model_path = Path(str(model.get("base_path", "")))
    if require_bp1_assets and not model_path.is_dir():
        raise ValueError(f"Base model not found: {model_path}")
    if model_path.name != model.get("snapshot_revision"):
        raise ValueError("Base model snapshot revision mismatch")

    data = _mapping(config, "data")
    checked_data = {}
    if require_bp1_assets:
        for name in ("bp1_manifest", "train", "dev", "challenge", "forbidden_gold"):
            contract = _mapping(data, name)
            path = repo_root / str(contract.get("path", ""))
            if not path.is_file():
                raise ValueError(f"Missing BP2 input: {path}")
            actual_hash = file_sha256(path)
            if actual_hash != contract.get("sha256"):
                raise ValueError(f"{name} SHA256 mismatch")
            if name in {"train", "dev", "challenge"}:
                rows = _jsonl_count(path)
                if rows != contract.get("rows"):
                    raise ValueError(f"{name} row count mismatch")
            checked_data[name] = {"path": str(path), "sha256": actual_hash}
        bp1_manifest = json.loads(
            (repo_root / data["bp1_manifest"]["path"]).read_text(encoding="utf-8")
        )
        if (
            bp1_manifest.get("dataset_version") != pipeline["dataset_version"]
            or bp1_manifest.get("target_contract_version")
            != pipeline["target_contract_version"]
            or bp1_manifest.get("frozen_gold_overlap") != 0
        ):
            raise ValueError("Accepted BP1 manifest contract mismatch")

    runtime = _mapping(config, "runtime")
    env = _mapping(runtime, "ENV")
    devices = [item for item in str(env.get("CUDA_VISIBLE_DEVICES", "")).split(",") if item]
    if not devices or int(env.get("NPROC_PER_NODE", 0)) != len(devices):
        raise ValueError("CUDA_VISIBLE_DEVICES and NPROC_PER_NODE mismatch")
    evaluation = _mapping(config, "evaluation")
    if (
        evaluation.get("selection_split") != "teacher_dev"
        or evaluation.get("diagnostic_split") != "challenge"
        or evaluation.get("forbidden_selection_split") != "frozen_human_gold"
    ):
        raise ValueError("BP2 evaluation split policy drift")

    stages = _mapping(config, "stages")
    if set(stages) != set(STAGES):
        raise ValueError("BP2 must define exactly sft, dpo, and grpo")
    for stage in STAGES:
        stage_config = _mapping(stages, stage)
        if stage_config.get("entrypoint") not in ENTRYPOINTS[stage]:
            raise ValueError(f"{stage} entrypoint mismatch")
        if stage == "dpo" and stage_config.get("rlhf_type", "dpo") != "dpo":
            raise ValueError("dpo.rlhf_type must be dpo")
        if stage == "grpo" and stage_config.get("rlhf_type") != "grpo":
            raise ValueError("grpo.rlhf_type must be grpo")
        for key in ("model_source", "dataset", "val_dataset", "output_dir", "authorization_file"):
            if not isinstance(stage_config.get(key), str) or not stage_config[key]:
                raise ValueError(f"{stage}.{key} must be non-empty")
            if Path(stage_config[key]).is_absolute():
                raise ValueError(f"{stage}.{key} must be repository-relative")
        dependencies = stage_config.get("dependencies")
        if not isinstance(dependencies, list):
            raise ValueError(f"{stage}.dependencies must be a list")
        _mapping(stage_config, "args")
        per_device_batch = int(stage_config["args"].get("per_device_train_batch_size", 0))
        accumulation = int(stage_config["args"].get("gradient_accumulation_steps", 0))
        if len(devices) * per_device_batch * accumulation != 32:
            raise ValueError(f"{stage} effective global batch size must remain 32")
        if stage == "grpo" and stage_config.get(
            "enforce_generation_batch_divisibility", False
        ):
            num_generations = int(stage_config["args"].get("num_generations", 0))
            per_device_eval_batch = int(
                stage_config["args"].get("per_device_eval_batch_size", 0)
            )
            if num_generations <= 0:
                raise ValueError("GRPO num_generations must be a positive integer")
            global_train_batch = len(devices) * per_device_batch
            if global_train_batch % num_generations != 0:
                raise ValueError(
                    "GRPO global train batch size must be divisible by "
                    "num_generations"
                )
            global_eval_batch = len(devices) * per_device_eval_batch
            if (
                per_device_eval_batch <= 0
                or global_eval_batch % num_generations != 0
            ):
                raise ValueError(
                    "GRPO global eval batch size must be divisible by "
                    "num_generations"
                )
    if stages["sft"]["dataset"] != data["train"]["path"]:
        raise ValueError("SFT must consume accepted BP1 train")
    if stages["sft"]["val_dataset"] != data["dev"]["path"]:
        raise ValueError("SFT must consume accepted BP1 dev")
    forbidden_gold = data["forbidden_gold"]["path"]
    if any(
        forbidden_gold
        in {stages[stage]["dataset"], stages[stage]["val_dataset"]}
        for stage in STAGES
    ):
        raise ValueError("Frozen human gold cannot be a training or validation dataset")
    if stages["sft"]["dependencies"]:
        raise ValueError("SFT engineering stage must not depend on DPO/GRPO")
    if not stages["dpo"]["dependencies"] or not stages["grpo"]["dependencies"]:
        raise ValueError("DPO/GRPO must remain dependency-gated")
    return {
        "pipeline_id": pipeline["id"],
        "dataset_version": pipeline["dataset_version"],
        "target_contract_version": pipeline["target_contract_version"],
        "checked_data": checked_data,
        "visible_device_count": len(devices),
    }


def _cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    raise ValueError(f"Unsupported CLI value: {type(value).__name__}")


def resolve_model(
    config: Mapping[str, Any],
    stage: str,
    authorization: Mapping[str, Any] | None,
) -> str:
    source = config["stages"][stage]["model_source"]
    if source in {
        "base",
        "base_with_sft_adapters",
        "base_with_upstream_adapters",
    }:
        return str(config["model"]["base_path"])
    if authorization and isinstance(authorization.get("model_path"), str):
        return authorization["model_path"]
    return (
        "<AUTHORIZED_SFT_CHECKPOINT>"
        if source == "accepted_sft_checkpoint"
        else "<AUTHORIZED_UPSTREAM_CHECKPOINT>"
    )


def build_stage_command(
    config: Mapping[str, Any],
    stage: str,
    *,
    authorization: Mapping[str, Any] | None = None,
    resume_from: str | None = None,
) -> Tuple[list[str], Dict[str, str]]:
    if stage not in STAGES:
        raise ValueError(f"Unknown BP2 stage: {stage}")
    stage_config = config["stages"][stage]
    executable = "rlhf" if stage == "dpo" else stage_config["entrypoint"]
    command = ["swift", executable]
    values: Dict[str, Any] = {
        "model": resolve_model(config, stage, authorization),
        "dataset": stage_config["dataset"],
        "val_dataset": stage_config["val_dataset"],
        "output_dir": stage_config["output_dir"],
        "seed": config["runtime"]["seed"],
        "torch_dtype": config["runtime"]["torch_dtype"],
        "report_to": config["runtime"]["report_to"],
        **stage_config["args"],
    }
    if (
        stage in {"dpo", "grpo"}
        and stage_config["model_source"]
        in {"base_with_sft_adapters", "base_with_upstream_adapters"}
    ):
        if not authorization or not authorization.get("model_path"):
            adapter_path = (
                "<AUTHORIZED_SFT_CHECKPOINT>"
                if stage == "dpo"
                else "<AUTHORIZED_UPSTREAM_CHECKPOINT>"
            )
        else:
            adapter_path = authorization["model_path"]
        values["adapters"] = adapter_path
        values["ref_adapters"] = adapter_path
    if stage in {"dpo", "grpo"}:
        values["rlhf_type"] = stage_config.get("rlhf_type", stage)
    if resume_from:
        values["resume_from_checkpoint"] = resume_from
    for key in sorted(values):
        command.extend([f"--{key}", _cli_value(values[key])])
    env = {
        str(key): _cli_value(value)
        for key, value in sorted(config["runtime"]["ENV"].items())
    }
    return command, env


def _is_reliability_authorization(
    stage: str,
    authorization: Mapping[str, Any],
) -> bool:
    return (
        stage == "dpo"
        and authorization.get("authorization_version")
        == "bp4-reliability-repair-active-v2"
    )


def _is_e4_grpo_authorization(
    stage: str,
    authorization: Mapping[str, Any],
) -> bool:
    return (
        stage == "grpo"
        and authorization.get("authorization_version")
        == "bp4-e4-grpo-from-dpo-active-v1"
    )


def _valid_two_level_evidence(
    evidence: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> bool:
    research_checks = evidence.get("research_progression_checks")
    valid = (
        evidence.get("status") == "PASS"
        and evidence.get("research_progression_decision")
        == "ACCEPT_SFT_RESEARCH_PROGRESSION"
        and isinstance(research_checks, dict)
        and bool(research_checks)
        and all(research_checks.values())
        and evidence.get("deployment_decision") == "REJECT_SFT_DEPLOYMENT"
    )
    if not _is_reliability_authorization("dpo", authorization):
        return valid
    return valid and (
        authorization.get("upstream_research_decision")
        == "ACCEPT_SFT_RESEARCH_PROGRESSION"
        and evidence.get("research_progression_decision")
        == authorization["upstream_research_decision"]
        and evidence.get("paired_packet_sha256")
        == authorization.get("upstream_paired_packet_sha256")
        and evidence.get("legacy_acceptance_sha256")
        == authorization.get("upstream_legacy_acceptance_sha256")
        and evidence.get("candidate_checkpoint") == authorization.get("model_path")
    )


def _valid_e4_grpo_evidence(
    evidence: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> bool:
    checks = evidence.get("checks")
    return (
        evidence.get("report_version")
        == "bp4-manual-research-progression-amendment-v1"
        and evidence.get("status") == "PASS"
        and evidence.get("decision")
        == "ACCEPT_DPO_AS_GRPO_RESEARCH_UPSTREAM"
        and isinstance(checks, dict)
        and bool(checks)
        and all(checks.values())
        and evidence.get("historical_research_decision")
        == "REJECT_DPO_RESEARCH_PROGRESSION"
        and evidence.get("historical_deployment_decision")
        == "REJECT_DPO_DEPLOYMENT"
        and evidence.get("candidate_checkpoint")
        == authorization.get("model_path")
        and evidence.get("paired_packet_sha256")
        == authorization.get("upstream_paired_packet_sha256")
        and evidence.get("historical_gate_sha256")
        == authorization.get("upstream_historical_gate_sha256")
    )


def _e4_dataset_hash(
    authorization: Mapping[str, Any],
    stage_config: Mapping[str, Any],
    data_key: str,
    repo_root: Path,
) -> str | None:
    source_value = authorization.get("dataset_hash_source_path")
    if not isinstance(source_value, str) or not source_value:
        return None
    source_path = Path(source_value)
    if not source_path.is_absolute():
        source_path = repo_root / source_path
    if not source_path.is_file():
        return None
    source = json.loads(source_path.read_text(encoding="utf-8"))
    file_key = "grpo_train" if data_key == "dataset" else "grpo_dev"
    file_spec = source.get("files", {}).get(file_key, {})
    expected_rows = 2360 if data_key == "dataset" else 331
    if (
        source.get("status") != "PASS"
        or not isinstance(source.get("checks"), dict)
        or not source["checks"]
        or not all(source["checks"].values())
        or source.get("counts", {}).get(file_key) != expected_rows
        or file_spec.get("path") != stage_config[data_key]
    ):
        return None
    value = file_spec.get("sha256")
    return value if isinstance(value, str) and len(value) == 64 else None


def read_authorization(
    config: Mapping[str, Any],
    stage: str,
    repo_root: Path,
    *,
    config_sha256: str,
) -> Tuple[Mapping[str, Any] | None, list[str]]:
    stage_config = config["stages"][stage]
    blockers = []
    authorization_path = repo_root / stage_config["authorization_file"]
    authorization = None
    if not authorization_path.is_file():
        blockers.append(f"missing_authorization:{stage_config['authorization_file']}")
    else:
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
        if authorization.get("decision") != AUTH_DECISIONS[stage]:
            blockers.append("authorization_decision_mismatch")
        if authorization.get("activation_status") != "ACTIVE_APPROVED":
            blockers.append("authorization_not_active")
        if authorization.get("stage") != stage:
            blockers.append("authorization_stage_mismatch")
        if authorization.get("config_sha256") != config_sha256:
            blockers.append("authorization_config_hash_mismatch")
    for dependency in stage_config["dependencies"]:
        path = repo_root / dependency
        if not path.is_file():
            blockers.append(f"missing_dependency:{dependency}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            blockers.append(f"dependency_not_accepted:{dependency}")
            continue
        if (
            authorization
            and _is_e4_grpo_authorization(stage, authorization)
            and (
                not isinstance(payload.get("checks"), dict)
                or not payload["checks"]
                or not all(payload["checks"].values())
            )
        ):
            blockers.append(f"dependency_checks_failed:{dependency}")
            continue
        if dependency.startswith("reports/model_gates/"):
            if payload.get("report_version") == "bp4-two-level-gate-amendment-v1":
                decision = payload.get("research_progression_decision")
                checks = payload.get("research_progression_checks")
                accepted_model_gate = (
                    decision == "ACCEPT_SFT_RESEARCH_PROGRESSION"
                    and isinstance(checks, dict)
                    and bool(checks)
                    and all(checks.values())
                )
            else:
                decision = str(payload.get("decision", ""))
                checks = payload.get("checks")
                accepted_model_gate = (
                    decision.startswith("ACCEPT_")
                    and isinstance(checks, dict)
                    and bool(checks)
                    and all(checks.values())
                )
            if not accepted_model_gate:
                blockers.append(f"dependency_not_accepted:{dependency}")
    for data_key in ("dataset", "val_dataset"):
        path = repo_root / stage_config[data_key]
        if not path.is_file():
            blockers.append(f"missing_{data_key}:{stage_config[data_key]}")
        elif authorization:
            if _is_e4_grpo_authorization(stage, authorization):
                expected_hash = _e4_dataset_hash(
                    authorization, stage_config, data_key, repo_root
                )
                if expected_hash is None:
                    blockers.append(f"{data_key}_hash_source_invalid")
                    continue
            else:
                expected_hash = authorization.get(f"{data_key}_sha256")
            if file_sha256(path) != expected_hash:
                blockers.append(f"{data_key}_authorization_hash_mismatch")
    if authorization:
        if stage == "sft":
            evidence_path_value = authorization.get("base_packet_path")
            evidence_hash = authorization.get("base_packet_sha256")
            evidence_kind = "base_packet"
        else:
            evidence_path_value = authorization.get("upstream_acceptance_path")
            evidence_hash = authorization.get("upstream_acceptance_sha256")
            evidence_kind = "upstream_acceptance"
        evidence_path = (
            Path(str(evidence_path_value))
            if evidence_path_value
            else None
        )
        if evidence_path is not None and not evidence_path.is_absolute():
            evidence_path = repo_root / evidence_path
        if evidence_path is None or not evidence_path.is_file():
            blockers.append(f"missing_authorization_evidence:{evidence_kind}")
        elif (
            not _is_reliability_authorization(stage, authorization)
            and file_sha256(evidence_path) != evidence_hash
        ):
            blockers.append(f"authorization_evidence_hash_mismatch:{evidence_kind}")
        else:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            if stage == "sft":
                valid_evidence = (
                    evidence.get("packet_version") == "bp4-base-evaluation-v1"
                    and evidence.get("status") == "PASS"
                    and evidence.get("dev_rows") == 331
                    and evidence.get("challenge_rows") == 140
                    and evidence.get("human_gold_used") is False
                )
            else:
                allowed = (
                    {"ACCEPT_SFT_CANDIDATE"}
                    if stage == "dpo"
                    else {"ACCEPT_SFT_CANDIDATE", "ACCEPT_DPO_CANDIDATE"}
                )
                if _is_e4_grpo_authorization(stage, authorization):
                    valid_evidence = _valid_e4_grpo_evidence(
                        evidence, authorization
                    )
                elif (
                    stage == "dpo"
                    and evidence.get("report_version")
                    == "bp4-two-level-gate-amendment-v1"
                ):
                    valid_evidence = _valid_two_level_evidence(
                        evidence, authorization
                    )
                else:
                    valid_evidence = (
                        evidence.get("report_version")
                        == "bp4-stage-acceptance-v1"
                        and evidence.get("status") == "PASS"
                        and evidence.get("decision") in allowed
                        and isinstance(evidence.get("checks"), dict)
                        and bool(evidence["checks"])
                        and all(evidence["checks"].values())
                    )
            if not valid_evidence:
                blockers.append(f"authorization_evidence_not_accepted:{evidence_kind}")
    if authorization and stage_config["model_source"] != "base":
        authorized_model = Path(str(authorization.get("model_path", "")))
        if not authorized_model.is_dir():
            blockers.append("authorized_model_path_missing")
        elif _is_e4_grpo_authorization(stage, authorization):
            trainer_state = authorized_model / "trainer_state.json"
            adapter_config = authorized_model / "adapter_config.json"
            adapter_weights = list(authorized_model.glob("adapter_model.*"))
            if not trainer_state.is_file():
                blockers.append("authorized_upstream_trainer_state_missing")
            elif file_sha256(trainer_state) != authorization.get(
                "upstream_trainer_state_sha256"
            ):
                blockers.append("authorized_upstream_trainer_state_hash_mismatch")
            if not adapter_config.is_file() or not adapter_weights:
                blockers.append("authorized_upstream_adapter_artifacts_missing")
    if stage == "grpo":
        plugin = repo_root / stage_config["args"]["external_plugins"]
        if not plugin.is_file():
            blockers.append(
                f"missing_reward_plugin:{stage_config['args']['external_plugins']}"
            )
        elif (
            authorization
            and _is_e4_grpo_authorization(stage, authorization)
            and file_sha256(plugin)
            != authorization.get("reward_plugin_sha256")
        ):
            blockers.append("reward_plugin_authorization_hash_mismatch")
    return authorization, sorted(blockers)


def validate_resume_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.is_dir():
        raise ValueError(f"Resume checkpoint is not a directory: {path}")
    state_file = path / "trainer_state.json"
    if not state_file.is_file():
        raise ValueError("Resume checkpoint lacks trainer_state.json")
    state = json.loads(state_file.read_text(encoding="utf-8"))
    global_step = state.get("global_step")
    if not isinstance(global_step, int) or global_step < 1:
        raise ValueError("Invalid checkpoint global_step")
    return {
        "checkpoint": str(path),
        "trainer_state_sha256": file_sha256(state_file),
        "global_step": global_step,
    }


def inspect_artifacts(output_dir: Path) -> Dict[str, Any]:
    checkpoints = [
        path
        for pattern in ("checkpoint-*", "*/checkpoint-*")
        for path in output_dir.glob(pattern)
        if path.is_dir()
    ]
    valid_by_run: Dict[Path, list[Dict[str, Any]]] = {}
    for checkpoint in checkpoints:
        try:
            report = validate_resume_checkpoint(checkpoint)
        except ValueError:
            continue
        report["trainer_state_mtime_ns"] = (
            checkpoint / "trainer_state.json"
        ).stat().st_mtime_ns
        valid_by_run.setdefault(checkpoint.parent, []).append(report)
    if not valid_by_run:
        raise ValueError(f"No valid checkpoints in {output_dir}")
    selected_run = max(
        valid_by_run,
        key=lambda run: max(
            item["trainer_state_mtime_ns"] for item in valid_by_run[run]
        ),
    )
    valid = sorted(
        valid_by_run[selected_run],
        key=lambda item: item["global_step"],
    )
    state = json.loads(
        (Path(valid[-1]["checkpoint"]) / "trainer_state.json").read_text(
            encoding="utf-8"
        )
    )
    best_model_checkpoint = state.get("best_model_checkpoint")
    if isinstance(best_model_checkpoint, str) and best_model_checkpoint:
        best_path = Path(best_model_checkpoint)
        if not best_path.is_absolute() and (selected_run / best_path).is_dir():
            best_model_checkpoint = str(selected_run / best_path)
    latest_checkpoint = dict(valid[-1])
    latest_checkpoint.pop("trainer_state_mtime_ns", None)
    return {
        "output_dir": str(output_dir),
        "selected_run_dir": str(selected_run),
        "discovered_run_count": len(valid_by_run),
        "run_selection": "most_recent_valid_trainer_state",
        "checkpoint_count": len(valid),
        "latest_checkpoint": latest_checkpoint,
        "best_model_checkpoint": best_model_checkpoint,
        "best_metric": state.get("best_metric"),
        "log_history_rows": len(state.get("log_history") or []),
    }


def evaluation_task(
    config: Mapping[str, Any],
    stage: str,
    model_path: str,
) -> Dict[str, Any]:
    return {
        "task_version": "bp2-unified-evaluation-task-v1",
        "stage": stage,
        "model_path": model_path,
        "dataset_version": config["pipeline"]["dataset_version"],
        "target_contract_version": config["pipeline"]["target_contract_version"],
        "selection": {
            "split": config["evaluation"]["selection_split"],
            "path": config["data"]["dev"]["path"],
        },
        "diagnostic": {
            "split": config["evaluation"]["diagnostic_split"],
            "path": config["data"]["challenge"]["path"],
        },
        "forbidden_for_selection": config["evaluation"]["forbidden_selection_split"],
        "metrics": config["evaluation"]["metrics"],
    }


def prepare_package(
    config_path: Path,
    output_dir: Path,
    repo_root: Path,
) -> Dict[str, Any]:
    config = load_config(config_path)
    contract = validate_config(config, repo_root, require_bp1_assets=True)
    config_sha256 = file_sha256(config_path)
    stage_plan = {}
    commands = {}
    checks = {
        "bp1_assets_valid": True,
        "all_stage_commands_build": True,
        "sft_has_no_stage_dependencies": not config["stages"]["sft"]["dependencies"],
        "dpo_is_dependency_gated": bool(config["stages"]["dpo"]["dependencies"]),
        "grpo_is_dependency_gated": bool(config["stages"]["grpo"]["dependencies"]),
        "human_gold_not_used_by_stages": all(
            config["data"]["forbidden_gold"]["path"]
            not in {
                config["stages"][stage]["dataset"],
                config["stages"][stage]["val_dataset"],
            }
            for stage in STAGES
        ),
    }
    for stage in STAGES:
        authorization, blockers = read_authorization(
            config, stage, repo_root, config_sha256=config_sha256
        )
        command, env = build_stage_command(
            config, stage, authorization=authorization
        )
        commands[stage] = {"command": command, "env": env}
        stage_plan[stage] = {
            "engineering_status": "READY",
            "execution_status": "BLOCKED" if blockers else "AUTHORIZED",
            "blockers": blockers,
            "authorization_file": config["stages"][stage]["authorization_file"],
            "dependencies": config["stages"][stage]["dependencies"],
        }
    evaluation_tasks = {
        stage: evaluation_task(config, stage, f"<{stage.upper()}_CHECKPOINT>")
        for stage in STAGES
    }
    checks["evaluation_tasks_cover_dev_and_challenge"] = all(
        task["selection"]["path"] == config["data"]["dev"]["path"]
        and task["diagnostic"]["path"] == config["data"]["challenge"]["path"]
        and task["forbidden_for_selection"] == "frozen_human_gold"
        for task in evaluation_tasks.values()
    )
    checks["sft_execution_is_authorization_gated"] = any(
        item.startswith("missing_authorization:") for item in stage_plan["sft"]["blockers"]
    )
    checks["dpo_execution_is_currently_blocked"] = bool(stage_plan["dpo"]["blockers"])
    checks["grpo_execution_is_currently_blocked"] = bool(stage_plan["grpo"]["blockers"])
    if not all(checks.values()):
        raise ValueError(f"BP2 engineering checks failed: {checks}")

    output_dir.mkdir(parents=True, exist_ok=True)
    quickstart = "\n".join(
        [
            "# BP2 Quickstart",
            "",
            "Prepare/validate the engineering package:",
            "",
            "```bash",
            "bash scripts/run_bp2_post_training_pipeline.sh",
            "```",
            "",
            "Inspect a stage without training:",
            "",
            "```bash",
            "python -m scripts.bp2_pipeline launch --config "
            "configs/experiments/bp2_post_training_pipeline_v1.yaml "
            "--stage sft --dry-run",
            "```",
            "",
            "Resume and real execution require a valid checkpoint, the stage-specific "
            "authorization file, and every declared dependency.",
            "",
        ]
    )
    quickstart_path = output_dir / "QUICKSTART.md"
    quickstart_path.write_text(quickstart, encoding="utf-8")
    evaluation_tasks_path = output_dir / "evaluation_tasks.json"
    evaluation_tasks_path.write_text(
        json.dumps(evaluation_tasks, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "manifest_version": PIPELINE_VERSION,
        "status": "PASS",
        "config": {"path": str(config_path), "sha256": config_sha256},
        "contract": contract,
        "source": _source_lineage(repo_root),
        "git": _git_lineage(repo_root),
        "environment": _environment_lineage(),
        "stage_plan": stage_plan,
        "commands": commands,
        "evaluation_tasks": evaluation_tasks,
        "documentation": {
            "quickstart": {
                "path": str(quickstart_path),
                "sha256": file_sha256(quickstart_path),
            },
            "evaluation_tasks": {
                "path": str(evaluation_tasks_path),
                "sha256": file_sha256(evaluation_tasks_path),
            },
        },
        "checks": checks,
    }
    manifest_path = output_dir / "bp2_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    gate = {
        "report_version": PIPELINE_VERSION,
        "status": "PASS",
        "decision": "ACCEPT_BP2_ENGINEERING",
        "checks": checks,
        "failed_checks": [],
        "stage_plan": stage_plan,
        "manifest_sha256": file_sha256(manifest_path),
        "next_work_package": "BP3_REWARD_V2_AND_PREFERENCE_DATA",
        "authorization": (
            "build_bp3; no_training_without_stage_specific_authorization"
        ),
    }
    gate_path = output_dir / "bp2_gate.json"
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    feedback = {
        "packet_version": "bp2-feedback-v1",
        "decision": gate["decision"],
        "next_work_package": gate["next_work_package"],
        "authorization": gate["authorization"],
        "failed_checks": [],
        "manifest_sha256": gate["manifest_sha256"],
        "gate_sha256": file_sha256(gate_path),
        "stage_plan": stage_plan,
        "return_files": [
            "new_plan/logs/bp2_post_training_pipeline.log",
            "reports/bp2_post_training_pipeline/bp2_manifest.json",
            "reports/bp2_post_training_pipeline/bp2_gate.json",
            "reports/bp2_post_training_pipeline/bp2_feedback.json",
            "reports/bp2_post_training_pipeline/evaluation_tasks.json",
            "reports/bp2_post_training_pipeline/QUICKSTART.md",
        ],
    }
    feedback_path = output_dir / "bp2_feedback.json"
    feedback_path.write_text(
        json.dumps(feedback, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    emit(
        "bp2_gate_complete",
        status="PASS",
        decision=gate["decision"],
        stage_plan=stage_plan,
        manifest_file=str(manifest_path),
        manifest_sha256=file_sha256(manifest_path),
        gate_file=str(gate_path),
        gate_sha256=file_sha256(gate_path),
        feedback_file=str(feedback_path),
        feedback_sha256=file_sha256(feedback_path),
        next_work_package=gate["next_work_package"],
    )
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/bp2_post_training_pipeline"),
    )
    prepare.add_argument("--repo-root", type=Path, default=Path("."))

    launch = subparsers.add_parser("launch")
    launch.add_argument("--config", type=Path, required=True)
    launch.add_argument("--stage", choices=STAGES, required=True)
    launch.add_argument("--repo-root", type=Path, default=Path("."))
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument(
        "--require-authorized",
        action="store_true",
        help="Fail dry-run when authorization or dependency blockers remain.",
    )
    launch.add_argument("--resume-from", type=Path)

    artifacts = subparsers.add_parser("inspect-artifacts")
    artifacts.add_argument("--output-dir", type=Path, required=True)
    artifacts.add_argument("--report-file", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare_package(args.config, args.output_dir, args.repo_root.resolve())
            return 0
        if args.command == "inspect-artifacts":
            report = inspect_artifacts(args.output_dir)
            if args.report_file:
                args.report_file.parent.mkdir(parents=True, exist_ok=True)
                args.report_file.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            emit("bp2_artifact_check", status="PASS", report=report)
            return 0

        config = load_config(args.config)
        validate_config(
            config, args.repo_root.resolve(), require_bp1_assets=not args.dry_run
        )
        authorization, blockers = read_authorization(
            config,
            args.stage,
            args.repo_root.resolve(),
            config_sha256=file_sha256(args.config),
        )
        resume = None
        if args.resume_from:
            resume = validate_resume_checkpoint(args.resume_from)["checkpoint"]
        command, env = build_stage_command(
            config,
            args.stage,
            authorization=authorization,
            resume_from=resume,
        )
        emit(
            "bp2_stage_launch",
            status=(
                "BLOCKED"
                if blockers and (args.require_authorized or not args.dry_run)
                else "PASS"
                if args.dry_run
                else "AUTHORIZED"
            ),
            stage=args.stage,
            dry_run=args.dry_run,
            blockers=blockers,
            command=command,
            env=env,
        )
        if args.dry_run:
            return 2 if args.require_authorized and blockers else 0
        if blockers:
            return 2
        process_env = os.environ.copy()
        process_env.update(env)
        return subprocess.run(command, env=process_env, check=False).returncode
    except Exception as exc:
        emit(
            "bp2_pipeline",
            status="FAIL",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
