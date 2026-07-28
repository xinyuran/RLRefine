"""Train-output evaluation and controlled acceptance packaging for BP4 SFT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from core.target_contract import TARGET_CONTRACT_VERSION
from evaluation.bp2_evaluator import evaluate_with_slices, prepare_reference
from evaluation.bp4_acceptance import (
    authorization_candidate,
    decide_stage,
)
from evaluation.bp4_base_v2 import (
    _run_split,
    _validate_prediction_metadata,
    prepare_inference_rows,
)
from evaluation.bp4_contract_telemetry import contract_aware_telemetry
from evaluation.bp4_stage_common import (
    metric_summary as _metric_summary,
    paired_bootstrap,
    sample_f1 as _sample_f1,
    select_best_checkpoint,
)
from evaluation.keyword_evaluator import (
    file_sha256,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from scripts.bp2_pipeline import load_config as load_bp2_config, read_authorization
from scripts.bp4_matrix import load_config, validate_dependencies
from scripts.prepare_experiment_manifest import (
    _environment_lineage,
    _git_lineage,
    _source_lineage,
)


OUTPUT_DIR = Path("reports/bp4_controlled_matrix/sft")
BASE_DIR = Path("reports/bp4_controlled_matrix/base_v2")
ARTIFACT_REPORT = OUTPUT_DIR / "sft_training_artifacts.json"
DEV_PREDICTIONS = OUTPUT_DIR / "dev_predictions.jsonl"
CHALLENGE_PREDICTIONS = OUTPUT_DIR / "challenge_predictions.jsonl"
DEV_EVALUATION = OUTPUT_DIR / "dev_evaluation.json"
CHALLENGE_EVALUATION = OUTPUT_DIR / "challenge_evaluation.json"
PAIRED_PACKET = OUTPUT_DIR / "sft_paired_evaluation_packet.json"
ACCEPTANCE_REPORT = OUTPUT_DIR / "sft_acceptance.json"
MANIFEST = OUTPUT_DIR / "sft_manifest.json"
GATE = OUTPUT_DIR / "sft_gate.json"
FEEDBACK = OUTPUT_DIR / "sft_feedback.json"


def emit(event: str, **payload: Any) -> None:
    print(
        json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


def _configs(root: Path, config_path: Path) -> tuple[Dict[str, Any], Dict[str, Any], Path]:
    config = load_config(config_path)
    validate_dependencies(root, config)
    bp2_path = root / config["pipeline"]["config"]
    return config, load_bp2_config(bp2_path), bp2_path


def preflight(root: Path, config_path: Path) -> Dict[str, Any]:
    config, bp2, bp2_path = _configs(root, config_path)
    authorization, blockers = read_authorization(
        bp2,
        "sft",
        root,
        config_sha256=file_sha256(bp2_path),
    )
    if blockers:
        raise ValueError(f"SFT authorization blockers: {blockers}")
    checks = {
        "authorization_active": authorization.get("activation_status")
        == "ACTIVE_APPROVED",
        "scope_is_sft_only": authorization.get("approved_scope")
        == "BP4_CONTROLLED_SFT_ONLY",
        "downstream_not_authorized": authorization.get(
            "downstream_training_authorized"
        )
        is False,
        "base_packet_bound": authorization.get("base_packet_sha256")
        == file_sha256(Path(authorization["base_packet_path"])),
        "human_gold_forbidden": config["evaluation"]["forbidden_selection"]
        == "frozen_human_gold",
    }
    if not all(checks.values()):
        raise ValueError(
            f"SFT preflight checks failed: "
            f"{[name for name, value in checks.items() if not value]}"
        )
    report = {
        "report_version": "bp4-sft-preflight-v1",
        "status": "PASS",
        "checks": checks,
        "authorization_sha256": file_sha256(
            root / bp2["stages"]["sft"]["authorization_file"]
        ),
        "authorized_scope": "BP4_CONTROLLED_SFT_ONLY",
        "dpo_authorized": False,
        "grpo_authorized": False,
    }
    write_json_atomic(root / OUTPUT_DIR / "sft_preflight.json", report)
    emit("bp4_sft_preflight", **report)
    return report


def run_inference(
    root: Path,
    config_path: Path,
    artifact_path: Path,
    *,
    batch_size: int,
    max_input_tokens: int,
) -> Dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BP4 SFT inference")
    preflight(root, config_path)
    config, bp2, _ = _configs(root, config_path)
    checkpoint = select_best_checkpoint(root, artifact_path)
    evaluation = config["evaluation"]
    dev_path = root / evaluation["selection"]["path"]
    challenge_path = root / evaluation["diagnostic"]["path"]
    dev_rows = prepare_inference_rows(read_jsonl(dev_path), "dev")
    challenge_rows = prepare_inference_rows(read_jsonl(challenge_path), "challenge")
    if len(dev_rows) != evaluation["selection"]["rows"]:
        raise ValueError("SFT dev row count mismatch")
    if len(challenge_rows) != evaluation["diagnostic"]["rows"]:
        raise ValueError("SFT challenge row count mismatch")

    output_dir = root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_manifest = {
        "manifest_version": "bp4-sft-inference-v1",
        "base_model_path": bp2["model"]["base_path"],
        "candidate_checkpoint": str(checkpoint),
        "candidate_checkpoint_trainer_state_sha256": file_sha256(
            checkpoint / "trainer_state.json"
        ),
        "selection_data_sha256": file_sha256(dev_path),
        "diagnostic_data_sha256": file_sha256(challenge_path),
        "decoding": evaluation["decoding"],
        "batch_size": batch_size,
        "max_input_tokens": max_input_tokens,
        "checkpoint_selection_metric": "eval_loss",
        "human_gold_used": False,
        "training_completed": True,
        "dpo_started": False,
        "grpo_started": False,
    }
    manifest_path = output_dir / "inference_manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != inference_manifest:
            raise ValueError("Existing SFT inference manifest does not match")
    else:
        write_json_atomic(manifest_path, inference_manifest)

    seed = evaluation["decoding"]["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        bp2["model"]["base_path"], trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        bp2["model"]["base_path"],
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(checkpoint)).cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    dev_result = _run_split(
        model=model,
        tokenizer=tokenizer,
        rows=dev_rows,
        prediction_path=root / DEV_PREDICTIONS,
        batch_size=batch_size,
        max_input_tokens=max_input_tokens,
        max_new_tokens=evaluation["decoding"]["max_new_tokens"],
    )
    challenge_result = _run_split(
        model=model,
        tokenizer=tokenizer,
        rows=challenge_rows,
        prediction_path=root / CHALLENGE_PREDICTIONS,
        batch_size=batch_size,
        max_input_tokens=max_input_tokens,
        max_new_tokens=evaluation["decoding"]["max_new_tokens"],
    )
    runtime = {
        "report_version": "bp4-sft-runtime-v1",
        "candidate_checkpoint": str(checkpoint),
        "dev": dev_result,
        "challenge": challenge_result,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "dpo_started": False,
        "grpo_started": False,
    }
    write_json_atomic(output_dir / "inference_runtime.json", runtime)
    emit(
        "bp4_sft_inference_complete",
        status="PASS",
        candidate_checkpoint=str(checkpoint),
        dev_rows=len(dev_rows),
        challenge_rows=len(challenge_rows),
        dpo_started=False,
        grpo_started=False,
    )
    return runtime


def package_evaluation(
    root: Path, config_path: Path, artifact_path: Path
) -> Dict[str, Any]:
    preflight_report = preflight(root, config_path)
    config, bp2, bp2_path = _configs(root, config_path)
    checkpoint = select_best_checkpoint(root, artifact_path)
    evaluation = config["evaluation"]
    dev_rows = read_jsonl(root / evaluation["selection"]["path"])
    challenge_rows = read_jsonl(root / evaluation["diagnostic"]["path"])
    dev_reference = prepare_reference(dev_rows)
    challenge_reference = prepare_reference(challenge_rows)
    base_dev = read_jsonl(root / BASE_DIR / "dev_predictions.jsonl")
    candidate_dev = read_jsonl(root / DEV_PREDICTIONS)
    candidate_challenge = read_jsonl(root / CHALLENGE_PREDICTIONS)
    _validate_prediction_metadata(base_dev + candidate_dev + candidate_challenge)
    if len(base_dev) != 331 or len(candidate_dev) != 331:
        raise ValueError("SFT paired dev coverage must be exactly 331")
    if len(candidate_challenge) != 140:
        raise ValueError("SFT challenge coverage must be exactly 140")

    base_report, _ = evaluate_with_slices(dev_reference, base_dev, "B1_V2")
    candidate_report, candidate_errors = evaluate_with_slices(
        dev_reference, candidate_dev, "E1_SFT"
    )
    challenge_report, challenge_errors = evaluate_with_slices(
        challenge_reference, candidate_challenge, "E1_SFT:challenge"
    )
    source_by_dev = {
        row["sample_id"]: row["normalized_text"] for row in dev_reference
    }
    source_by_challenge = {
        row["sample_id"]: row["normalized_text"] for row in challenge_reference
    }
    base_telemetry = contract_aware_telemetry(base_dev, source_by_dev)
    candidate_telemetry = contract_aware_telemetry(candidate_dev, source_by_dev)
    challenge_telemetry = contract_aware_telemetry(
        candidate_challenge, source_by_challenge
    )
    output_dir = root / OUTPUT_DIR
    write_json_atomic(root / DEV_EVALUATION, candidate_report)
    write_json_atomic(root / CHALLENGE_EVALUATION, challenge_report)
    write_jsonl_atomic(output_dir / "dev_errors.jsonl", candidate_errors)
    write_jsonl_atomic(output_dir / "challenge_errors.jsonl", challenge_errors)
    write_json_atomic(output_dir / "base_contract_telemetry.json", base_telemetry)
    write_json_atomic(
        output_dir / "candidate_contract_telemetry.json",
        {
            "telemetry_bundle_version": "bp4-sft-contract-telemetry-bundle-v1",
            "dev": candidate_telemetry,
            "challenge": challenge_telemetry,
            "challenge_usage": "diagnostic_only",
        },
    )
    max_new_tokens = evaluation["decoding"]["max_new_tokens"]
    baseline_metrics = _metric_summary(
        base_report, base_dev, base_telemetry, max_new_tokens
    )
    candidate_metrics = _metric_summary(
        candidate_report,
        candidate_dev,
        candidate_telemetry,
        max_new_tokens,
    )
    paired = paired_bootstrap(
        _sample_f1(dev_reference, base_dev),
        _sample_f1(dev_reference, candidate_dev),
        iterations=evaluation["bootstrap_iterations"],
        seed=evaluation["decoding"]["seed"],
    )
    packet = {
        "packet_version": "bp4-paired-evaluation-v1",
        "selection_split": "teacher_dev",
        "sample_count": 331,
        "human_gold_used": False,
        "challenge_usage": "diagnostic_only",
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "bootstrap_iterations": evaluation["bootstrap_iterations"],
        "bootstrap_seed": evaluation["decoding"]["seed"],
        "selection_data_sha256": file_sha256(
            root / evaluation["selection"]["path"]
        ),
        "baseline_predictions_sha256": file_sha256(
            root / BASE_DIR / "dev_predictions.jsonl"
        ),
        "candidate_predictions_sha256": file_sha256(root / DEV_PREDICTIONS),
        "baseline_model_id": "B1_V2",
        "candidate_model_id": "E1_SFT",
        "baseline_checkpoint": bp2["model"]["base_path"],
        "candidate_checkpoint": str(checkpoint),
        "baseline_stage": "base",
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "paired": paired,
        "challenge": {
            "sample_count": 140,
            "usage": "diagnostic_only",
            "metrics": _metric_summary(
                challenge_report,
                candidate_challenge,
                challenge_telemetry,
                max_new_tokens,
            ),
        },
    }
    write_json_atomic(root / PAIRED_PACKET, packet)
    acceptance = decide_stage(
        packet,
        config["acceptance"],
        stage="sft",
        candidate_checkpoint=str(checkpoint),
    )
    write_json_atomic(root / ACCEPTANCE_REPORT, acceptance)

    accepted = acceptance["decision"] == "ACCEPT_SFT_CANDIDATE"
    downstream_candidates = {}
    if accepted:
        model_gate = root / "reports/model_gates/bp2_sft_acceptance.json"
        write_json_atomic(model_gate, acceptance)
        for next_stage in ("dpo", "grpo"):
            stage_config = bp2["stages"][next_stage]
            candidate = authorization_candidate(
                acceptance,
                next_stage=next_stage,
                bp2_config_sha256=file_sha256(bp2_path),
                model_path=str(checkpoint),
                dataset_sha256=file_sha256(root / stage_config["dataset"]),
                val_dataset_sha256=file_sha256(root / stage_config["val_dataset"]),
            )
            candidate.update(
                {
                    "active_path": stage_config["authorization_file"],
                    "upstream_acceptance_path": str(
                        ACCEPTANCE_REPORT
                    ).replace("\\", "/"),
                    "upstream_acceptance_sha256": file_sha256(
                        root / ACCEPTANCE_REPORT
                    ),
                    "approved_scope": "CANDIDATE_ONLY_REQUIRES_NEW_HUMAN_APPROVAL",
                    "downstream_training_authorized": False,
                }
            )
            path = (
                root
                / "reports/bp4_controlled_matrix/authorization_candidates"
                / f"bp2_{next_stage}_ready.json"
            )
            write_json_atomic(path, candidate)
            downstream_candidates[next_stage] = {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": file_sha256(path),
                "activation_status": "CANDIDATE_NOT_ACTIVE",
            }

    output_paths = {
        "training_artifacts": artifact_path,
        "dev_predictions": root / DEV_PREDICTIONS,
        "challenge_predictions": root / CHALLENGE_PREDICTIONS,
        "dev_evaluation": root / DEV_EVALUATION,
        "challenge_evaluation": root / CHALLENGE_EVALUATION,
        "paired_packet": root / PAIRED_PACKET,
        "acceptance": root / ACCEPTANCE_REPORT,
        "base_contract_telemetry": output_dir / "base_contract_telemetry.json",
        "candidate_contract_telemetry": output_dir
        / "candidate_contract_telemetry.json",
    }
    checks = {
        "preflight_passed": preflight_report["status"] == "PASS",
        "best_checkpoint_selected_by_eval_loss": True,
        "dev_coverage_exact_331": len(candidate_dev) == 331,
        "challenge_coverage_exact_140": len(candidate_challenge) == 140,
        "human_gold_not_used": packet["human_gold_used"] is False,
        "challenge_diagnostic_only": packet["challenge_usage"]
        == "diagnostic_only",
        "contract_aware_hallucination_used": all(
            metrics["hallucination_metric_version"]
            == "bp4-contract-aware-telemetry-v1"
            for metrics in (baseline_metrics, candidate_metrics)
        ),
        "dpo_not_started": True,
        "grpo_not_started": True,
    }
    manifest = {
        "manifest_version": "bp4-sft-package-v1",
        "config_sha256": file_sha256(config_path),
        "bp2_config_sha256": file_sha256(bp2_path),
        "authorization_sha256": preflight_report["authorization_sha256"],
        "candidate_checkpoint": str(checkpoint),
        "checkpoint_selection_metric": "eval_loss",
        "outputs": {
            name: {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": file_sha256(path),
            }
            for name, path in output_paths.items()
        },
        "downstream_authorization_candidates": downstream_candidates,
        "checks": checks,
        "source": _source_lineage(root),
        "git": _git_lineage(root),
        "environment": _environment_lineage(),
        "human_gold_used": False,
        "dpo_started": False,
        "grpo_started": False,
    }
    write_json_atomic(root / MANIFEST, manifest)
    gate = {
        "report_version": "bp4-sft-package-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "decision": acceptance["decision"],
        "candidate_checkpoint": str(checkpoint),
        "acceptance_sha256": file_sha256(root / ACCEPTANCE_REPORT),
        "paired_packet_sha256": file_sha256(root / PAIRED_PACKET),
        "manifest_sha256": file_sha256(root / MANIFEST),
        "checks": checks,
        "acceptance_failed_checks": acceptance["failed_checks"],
        "next_execution": (
            "BP4_DPO_GRPO_AUTHORIZATION_REVIEW"
            if accepted
            else "STOP_POST_TRAINING_AT_BASE_V2"
        ),
        "downstream_training_authorized": False,
        "dpo_started": False,
        "grpo_started": False,
    }
    write_json_atomic(root / GATE, gate)
    return_files = [
        "new_plan/logs/bp4_controlled_sft.log",
        str(ARTIFACT_REPORT).replace("\\", "/"),
        str(DEV_PREDICTIONS).replace("\\", "/"),
        str(CHALLENGE_PREDICTIONS).replace("\\", "/"),
        str(DEV_EVALUATION).replace("\\", "/"),
        str(CHALLENGE_EVALUATION).replace("\\", "/"),
        str(PAIRED_PACKET).replace("\\", "/"),
        str(ACCEPTANCE_REPORT).replace("\\", "/"),
        str(MANIFEST).replace("\\", "/"),
        str(GATE).replace("\\", "/"),
        str(FEEDBACK).replace("\\", "/"),
        str(OUTPUT_DIR / "base_contract_telemetry.json").replace("\\", "/"),
        str(OUTPUT_DIR / "candidate_contract_telemetry.json").replace("\\", "/"),
    ]
    feedback = {
        "packet_version": "bp4-sft-feedback-v1",
        "decision": gate["decision"],
        "plain_language_result": (
            "SFT achieved a credible controlled gain; downstream stages remain "
            "inactive pending a separate approval."
            if accepted
            else "SFT did not clear every preregistered gate; keep Base v2 and "
            "do not start DPO or GRPO."
        ),
        "acceptance_failed_checks": gate["acceptance_failed_checks"],
        "next_execution": gate["next_execution"],
        "downstream_training_authorized": False,
        "return_files": return_files,
    }
    write_json_atomic(root / FEEDBACK, feedback)
    emit(
        "bp4_sft_gate_complete",
        status=gate["status"],
        decision=gate["decision"],
        next_execution=gate["next_execution"],
        downstream_training_authorized=False,
        return_file_count=len(return_files),
    )
    if gate["status"] != "PASS":
        raise ValueError("BP4 SFT package integrity checks failed")
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/bp4_controlled_model_matrix_v1.yaml"),
    )
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_REPORT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    infer = subparsers.add_parser("infer")
    infer.add_argument("--batch-size", type=int, default=4)
    infer.add_argument("--max-input-tokens", type=int, default=4096)
    subparsers.add_parser("package")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    artifact_path = (
        args.artifacts if args.artifacts.is_absolute() else root / args.artifacts
    )
    try:
        if args.command == "preflight":
            preflight(root, config_path)
        elif args.command == "infer":
            run_inference(
                root,
                config_path,
                artifact_path,
                batch_size=args.batch_size,
                max_input_tokens=args.max_input_tokens,
            )
        else:
            package_evaluation(root, config_path, artifact_path)
        return 0
    except Exception as exc:
        emit(
            "bp4_sft_v2",
            status="FAIL",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
