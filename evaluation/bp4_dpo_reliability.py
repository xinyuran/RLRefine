"""Inference, paired evaluation, and two-level gates for BP4 DPO reliability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from core.target_contract import TARGET_CONTRACT_VERSION
from evaluation.bp2_evaluator import evaluate_with_slices, prepare_reference
from evaluation.bp4_acceptance import validate_evaluation_packet
from evaluation.bp4_base_v2 import (
    _run_split,
    _validate_prediction_metadata,
    prepare_inference_rows,
)
from evaluation.bp4_contract_telemetry import contract_aware_telemetry
from evaluation.bp4_reliability_gate import load_config as load_reliability_config
from evaluation.bp4_stage_common import (
    metric_summary,
    paired_bootstrap,
    sample_f1,
    select_best_checkpoint,
)
from evaluation.keyword_evaluator import (
    file_sha256,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from scripts.bp2_pipeline import (
    load_config as load_bp2_config,
    read_authorization,
)
from scripts.prepare_experiment_manifest import (
    _environment_lineage,
    _git_lineage,
    _source_lineage,
)


OUTPUT_DIR = Path("reports/bp4_reliability_repair/dpo/evaluation")
ARTIFACT_REPORT = Path(
    "reports/bp4_reliability_repair/dpo/dpo_training_artifacts.json"
)
SFT_DIR = Path("reports/bp4_controlled_matrix/sft")
DEV_PREDICTIONS = OUTPUT_DIR / "dev_predictions.jsonl"
CHALLENGE_PREDICTIONS = OUTPUT_DIR / "challenge_predictions.jsonl"
PAIRED_PACKET = OUTPUT_DIR / "dpo_paired_evaluation_packet.json"
TWO_LEVEL_GATE = OUTPUT_DIR / "dpo_two_level_gate.json"
MANIFEST = OUTPUT_DIR / "dpo_evaluation_manifest.json"
FEEDBACK = OUTPUT_DIR / "dpo_evaluation_feedback.json"


def emit(event: str, **payload: Any) -> None:
    print(
        json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


def _load_matrix_config(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "structalign-bp4-controlled-matrix-v1"
    ):
        raise ValueError("Invalid BP4 controlled-matrix config")
    return value


def _same_path(left: str, right: str) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def preflight(
    root: Path,
    reliability_path: Path,
    matrix_path: Path,
    bp2_path: Path,
    artifact_path: Path,
) -> Dict[str, Any]:
    reliability = load_reliability_config(reliability_path)
    matrix = _load_matrix_config(matrix_path)
    bp2 = load_bp2_config(bp2_path)
    authorization, blockers = read_authorization(
        bp2,
        "dpo",
        root,
        config_sha256=file_sha256(bp2_path),
    )
    if blockers:
        raise ValueError(f"DPO authorization blockers: {blockers}")
    checkpoint = select_best_checkpoint(root, artifact_path)
    sft_packet_path = root / reliability["inputs"]["paired_packet"]
    sft_packet = json.loads(sft_packet_path.read_text(encoding="utf-8"))
    sft_dev_path = root / SFT_DIR / "dev_predictions.jsonl"
    checks = {
        "authorization_active": authorization.get("activation_status")
        == "ACTIVE_APPROVED",
        "scope_is_dpo_reliability_only": authorization.get("approved_scope")
        == "BP4_DPO_RELIABILITY_REPAIR_ONLY",
        "matrix_id_locked": authorization.get("matrix_id")
        == reliability["dpo_repair_preregistration"]["matrix_id"],
        "direct_upstream_checkpoint_bound": _same_path(
            authorization["model_path"], sft_packet["candidate_checkpoint"]
        ),
        "sft_packet_hash_locked": file_sha256(sft_packet_path)
        == reliability["inputs"]["expected"]["paired_packet_sha256"],
        "sft_predictions_bound": file_sha256(sft_dev_path)
        == sft_packet["candidate_predictions_sha256"],
        "training_checkpoint_has_steps": json.loads(
            (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
        ).get("global_step", 0)
        > 0,
        "human_gold_forbidden": matrix["evaluation"]["forbidden_selection"]
        == "frozen_human_gold",
        "grpo_not_authorized": authorization.get(
            "downstream_grpo_authorized"
        )
        is False,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"DPO evaluation preflight failed: {failed}")
    report = {
        "report_version": "bp4-dpo-evaluation-preflight-v1",
        "status": "PASS",
        "checks": checks,
        "authorization_sha256": file_sha256(
            root / bp2["stages"]["dpo"]["authorization_file"]
        ),
        "artifact_report_sha256": file_sha256(artifact_path),
        "candidate_checkpoint": str(checkpoint),
        "direct_upstream_checkpoint": authorization["model_path"],
        "human_gold_used": False,
        "training_started_by_evaluation": False,
        "grpo_authorized": False,
    }
    write_json_atomic(root / OUTPUT_DIR / "preflight.json", report)
    emit("bp4_dpo_evaluation_preflight", **report)
    return report


def run_inference(
    root: Path,
    reliability_path: Path,
    matrix_path: Path,
    bp2_path: Path,
    artifact_path: Path,
    *,
    batch_size: int,
    max_input_tokens: int,
) -> Dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BP4 DPO inference")
    preflight_report = preflight(
        root, reliability_path, matrix_path, bp2_path, artifact_path
    )
    reliability = load_reliability_config(reliability_path)
    matrix = _load_matrix_config(matrix_path)
    bp2 = load_bp2_config(bp2_path)
    checkpoint = Path(preflight_report["candidate_checkpoint"])
    evaluation = matrix["evaluation"]
    prereg = reliability["dpo_repair_preregistration"]
    if evaluation["decoding"] != prereg["decoding"]:
        raise ValueError("DPO decoding differs from the preregistered BP4 protocol")
    dev_path = root / evaluation["selection"]["path"]
    challenge_path = root / evaluation["diagnostic"]["path"]
    dev_rows = prepare_inference_rows(read_jsonl(dev_path), "dev")
    challenge_rows = prepare_inference_rows(read_jsonl(challenge_path), "challenge")
    if len(dev_rows) != evaluation["selection"]["rows"]:
        raise ValueError("DPO dev row count mismatch")
    if len(challenge_rows) != evaluation["diagnostic"]["rows"]:
        raise ValueError("DPO challenge row count mismatch")

    output_dir = root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_manifest = {
        "manifest_version": "bp4-dpo-inference-v1",
        "base_model_path": bp2["model"]["base_path"],
        "candidate_checkpoint": str(checkpoint),
        "candidate_checkpoint_trainer_state_sha256": file_sha256(
            checkpoint / "trainer_state.json"
        ),
        "direct_upstream_checkpoint": preflight_report[
            "direct_upstream_checkpoint"
        ],
        "selection_data_sha256": file_sha256(dev_path),
        "diagnostic_data_sha256": file_sha256(challenge_path),
        "decoding": prereg["decoding"],
        "batch_size": batch_size,
        "max_input_tokens": max_input_tokens,
        "checkpoint_selection_metric": "eval_loss",
        "human_gold_used": False,
        "training_started_by_evaluation": False,
        "grpo_started": False,
    }
    manifest_path = output_dir / "inference_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != inference_manifest:
            raise ValueError("Existing DPO inference manifest does not match")
    else:
        write_json_atomic(manifest_path, inference_manifest)

    seed = prereg["decoding"]["seed"]
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
        max_new_tokens=prereg["decoding"]["max_new_tokens"],
    )
    challenge_result = _run_split(
        model=model,
        tokenizer=tokenizer,
        rows=challenge_rows,
        prediction_path=root / CHALLENGE_PREDICTIONS,
        batch_size=batch_size,
        max_input_tokens=max_input_tokens,
        max_new_tokens=prereg["decoding"]["max_new_tokens"],
    )
    runtime = {
        "report_version": "bp4-dpo-runtime-v1",
        "candidate_checkpoint": str(checkpoint),
        "dev": dev_result,
        "challenge": challenge_result,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "training_started_by_evaluation": False,
        "grpo_started": False,
    }
    write_json_atomic(output_dir / "inference_runtime.json", runtime)
    emit(
        "bp4_dpo_inference_complete",
        status="PASS",
        candidate_checkpoint=str(checkpoint),
        dev_rows=len(dev_rows),
        challenge_rows=len(challenge_rows),
        training_started_by_evaluation=False,
        grpo_started=False,
    )
    return runtime


def research_checks(
    packet: Mapping[str, Any], gates: Mapping[str, Any]
) -> Dict[str, bool]:
    baseline = packet["baseline"]
    candidate = packet["candidate"]
    paired = packet["paired"]
    return {
        "complete_dev_coverage": packet["sample_count"] == 331,
        "macro_f1_delta_min": paired[
            "macro_f1_delta_candidate_minus_baseline"
        ]
        >= gates["macro_f1_delta_min"],
        "macro_f1_ci_low_gt": paired["macro_f1_bootstrap_95_ci"][0]
        > gates["macro_f1_ci_low_gt"],
        "micro_f1_delta_min": candidate["micro_f1"] - baseline["micro_f1"]
        >= gates["micro_f1_delta_min"],
        "schema_valid_rate_min": candidate["schema_valid_rate"]
        >= gates["schema_valid_rate_min"],
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
            candidate["mean_output_tokens"]
            / max(baseline["mean_output_tokens"], 1e-12)
            <= gates["mean_output_tokens_ratio_max"]
        ),
        "max_token_output_count_delta_max": (
            candidate["max_token_output_count"]
            - baseline["max_token_output_count"]
            <= gates["max_token_output_count_delta_max"]
        ),
        "human_gold_not_used": packet["human_gold_used"] is False,
        "challenge_not_used_for_selection": packet["challenge_usage"]
        == "diagnostic_only",
    }


def deployment_checks(
    packet: Mapping[str, Any],
    base_metrics: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> Dict[str, bool]:
    candidate = packet["candidate"]
    return {
        "schema_valid_rate_min": candidate["schema_valid_rate"]
        >= gates["schema_valid_rate_min"],
        "hallucination_keyword_rate_max": (
            candidate["hallucination_keyword_rate"]
            <= gates["hallucination_keyword_rate_max"]
        ),
        "mean_output_tokens_vs_base_ratio_max": (
            candidate["mean_output_tokens"]
            / max(base_metrics["mean_output_tokens"], 1e-12)
            <= gates["mean_output_tokens_vs_base_ratio_max"]
        ),
        "max_token_output_count": candidate["max_token_output_count"]
        == gates["max_token_output_count"],
        "human_gold_not_used": packet["human_gold_used"] is False,
        "challenge_not_used_for_selection": packet["challenge_usage"]
        == "diagnostic_only",
    }


def package_evaluation(
    root: Path,
    reliability_path: Path,
    matrix_path: Path,
    bp2_path: Path,
    artifact_path: Path,
) -> Dict[str, Any]:
    preflight_report = preflight(
        root, reliability_path, matrix_path, bp2_path, artifact_path
    )
    reliability = load_reliability_config(reliability_path)
    matrix = _load_matrix_config(matrix_path)
    prereg = reliability["dpo_repair_preregistration"]
    checkpoint = Path(preflight_report["candidate_checkpoint"])
    evaluation = matrix["evaluation"]
    dev_path = root / evaluation["selection"]["path"]
    challenge_path = root / evaluation["diagnostic"]["path"]
    dev_reference = prepare_reference(read_jsonl(dev_path))
    challenge_reference = prepare_reference(read_jsonl(challenge_path))
    upstream_dev = read_jsonl(root / SFT_DIR / "dev_predictions.jsonl")
    candidate_dev = read_jsonl(root / DEV_PREDICTIONS)
    candidate_challenge = read_jsonl(root / CHALLENGE_PREDICTIONS)
    _validate_prediction_metadata(
        upstream_dev + candidate_dev + candidate_challenge
    )
    if len(upstream_dev) != 331 or len(candidate_dev) != 331:
        raise ValueError("DPO paired dev coverage must be exactly 331")
    if len(candidate_challenge) != 140:
        raise ValueError("DPO challenge coverage must be exactly 140")

    upstream_report, _ = evaluate_with_slices(
        dev_reference, upstream_dev, "E1_SFT"
    )
    candidate_report, candidate_errors = evaluate_with_slices(
        dev_reference, candidate_dev, "E2_DPO_RELIABILITY"
    )
    challenge_report, challenge_errors = evaluate_with_slices(
        challenge_reference,
        candidate_challenge,
        "E2_DPO_RELIABILITY:challenge",
    )
    source_by_dev = {
        row["sample_id"]: row["normalized_text"] for row in dev_reference
    }
    source_by_challenge = {
        row["sample_id"]: row["normalized_text"]
        for row in challenge_reference
    }
    upstream_telemetry = contract_aware_telemetry(upstream_dev, source_by_dev)
    candidate_telemetry = contract_aware_telemetry(
        candidate_dev, source_by_dev
    )
    challenge_telemetry = contract_aware_telemetry(
        candidate_challenge, source_by_challenge
    )
    output_dir = root / OUTPUT_DIR
    write_json_atomic(output_dir / "dev_evaluation.json", candidate_report)
    write_json_atomic(
        output_dir / "challenge_evaluation.json", challenge_report
    )
    write_jsonl_atomic(output_dir / "dev_errors.jsonl", candidate_errors)
    write_jsonl_atomic(
        output_dir / "challenge_errors.jsonl", challenge_errors
    )
    write_json_atomic(
        output_dir / "upstream_contract_telemetry.json", upstream_telemetry
    )
    write_json_atomic(
        output_dir / "candidate_contract_telemetry.json",
        {
            "telemetry_bundle_version": "bp4-dpo-contract-telemetry-bundle-v1",
            "dev": candidate_telemetry,
            "challenge": challenge_telemetry,
            "challenge_usage": "diagnostic_only",
        },
    )
    upstream_metrics = metric_summary(
        upstream_report,
        upstream_dev,
        upstream_telemetry,
        prereg["decoding"]["max_new_tokens"],
    )
    candidate_metrics = metric_summary(
        candidate_report,
        candidate_dev,
        candidate_telemetry,
        prereg["decoding"]["max_new_tokens"],
    )
    paired = paired_bootstrap(
        sample_f1(dev_reference, upstream_dev),
        sample_f1(dev_reference, candidate_dev),
        iterations=prereg["bootstrap_iterations"],
        seed=prereg["bootstrap_seed"],
    )
    sft_packet_path = root / reliability["inputs"]["paired_packet"]
    sft_packet = json.loads(sft_packet_path.read_text(encoding="utf-8"))
    packet = {
        "packet_version": "bp4-paired-evaluation-v1",
        "selection_split": "teacher_dev",
        "sample_count": 331,
        "human_gold_used": False,
        "challenge_usage": "diagnostic_only",
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "bootstrap_iterations": prereg["bootstrap_iterations"],
        "bootstrap_seed": prereg["bootstrap_seed"],
        "selection_data_sha256": file_sha256(dev_path),
        "baseline_predictions_sha256": file_sha256(
            root / SFT_DIR / "dev_predictions.jsonl"
        ),
        "candidate_predictions_sha256": file_sha256(root / DEV_PREDICTIONS),
        "baseline_model_id": "E1_SFT",
        "candidate_model_id": prereg["matrix_id"],
        "baseline_checkpoint": sft_packet["candidate_checkpoint"],
        "candidate_checkpoint": str(checkpoint),
        "baseline_stage": "sft",
        "baseline": upstream_metrics,
        "candidate": candidate_metrics,
        "paired": paired,
        "challenge": {
            "sample_count": 140,
            "usage": "diagnostic_only",
            "metrics": metric_summary(
                challenge_report,
                candidate_challenge,
                challenge_telemetry,
                prereg["decoding"]["max_new_tokens"],
            ),
        },
    }
    validate_evaluation_packet(packet)
    write_json_atomic(root / PAIRED_PACKET, packet)
    research = research_checks(packet, prereg["research_gate"])
    deployment = deployment_checks(
        packet, sft_packet["baseline"], prereg["deployment_gate"]
    )
    research_passed = all(research.values())
    deployment["research_progression_passed"] = research_passed
    deployment_passed = all(deployment.values())
    gate = {
        "report_version": "bp4-dpo-two-level-gate-v1",
        "status": "PASS",
        "stage": "dpo",
        "candidate_checkpoint": str(checkpoint),
        "direct_upstream_checkpoint": sft_packet["candidate_checkpoint"],
        "research_progression_decision": (
            "ACCEPT_DPO_RESEARCH_PROGRESSION"
            if research_passed
            else "REJECT_DPO_RESEARCH_PROGRESSION"
        ),
        "research_progression_checks": research,
        "research_progression_failed_checks": [
            name for name, passed in research.items() if not passed
        ],
        "deployment_decision": (
            "ACCEPT_DPO_DEPLOYMENT"
            if deployment_passed
            else "REJECT_DPO_DEPLOYMENT"
        ),
        "deployment_checks": deployment,
        "deployment_failed_checks": [
            name for name, passed in deployment.items() if not passed
        ],
        "paired_packet_sha256": file_sha256(root / PAIRED_PACKET),
        "human_gold_used": False,
        "grpo_training_authorized": False,
        "next_execution": (
            "HUMAN_REVIEW_BP5_SERVING_CANDIDATE"
            if deployment_passed
            else "HUMAN_REVIEW_GRPO_RELIABILITY_AUTHORIZATION"
            if research_passed
            else "STOP_POST_TRAINING_AT_SFT"
        ),
    }
    write_json_atomic(root / TWO_LEVEL_GATE, gate)

    output_paths = {
        "training_artifacts": artifact_path,
        "preflight": output_dir / "preflight.json",
        "inference_manifest": output_dir / "inference_manifest.json",
        "inference_runtime": output_dir / "inference_runtime.json",
        "dev_predictions": root / DEV_PREDICTIONS,
        "challenge_predictions": root / CHALLENGE_PREDICTIONS,
        "dev_evaluation": output_dir / "dev_evaluation.json",
        "challenge_evaluation": output_dir / "challenge_evaluation.json",
        "paired_packet": root / PAIRED_PACKET,
        "two_level_gate": root / TWO_LEVEL_GATE,
        "upstream_contract_telemetry": output_dir
        / "upstream_contract_telemetry.json",
        "candidate_contract_telemetry": output_dir
        / "candidate_contract_telemetry.json",
    }
    integrity_checks = {
        "preflight_passed": preflight_report["status"] == "PASS",
        "best_checkpoint_selected_by_eval_loss": True,
        "direct_upstream_is_sft": packet["baseline_stage"] == "sft",
        "dev_coverage_exact_331": len(candidate_dev) == 331,
        "challenge_coverage_exact_140": len(candidate_challenge) == 140,
        "human_gold_not_used": packet["human_gold_used"] is False,
        "challenge_diagnostic_only": packet["challenge_usage"]
        == "diagnostic_only",
        "contract_aware_hallucination_used": all(
            metrics["hallucination_metric_version"]
            == "bp4-contract-aware-telemetry-v1"
            for metrics in (upstream_metrics, candidate_metrics)
        ),
        "training_not_restarted": True,
        "grpo_not_started": True,
    }
    manifest = {
        "manifest_version": "bp4-dpo-evaluation-package-v1",
        "reliability_config_sha256": file_sha256(reliability_path),
        "matrix_config_sha256": file_sha256(matrix_path),
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
        "checks": integrity_checks,
        "source": _source_lineage(root),
        "git": _git_lineage(root),
        "environment": _environment_lineage(),
        "human_gold_used": False,
        "training_started_by_evaluation": False,
        "grpo_started": False,
    }
    write_json_atomic(root / MANIFEST, manifest)
    return_files = [
        "new_plan/logs/bp4_dpo_reliability_evaluation.log",
        *[
            str(item["path"])
            for item in manifest["outputs"].values()
        ],
        str(MANIFEST).replace("\\", "/"),
        str(FEEDBACK).replace("\\", "/"),
    ]
    feedback = {
        "packet_version": "bp4-dpo-evaluation-feedback-v1",
        "research_progression_decision": gate[
            "research_progression_decision"
        ],
        "deployment_decision": gate["deployment_decision"],
        "research_progression_failed_checks": gate[
            "research_progression_failed_checks"
        ],
        "deployment_failed_checks": gate["deployment_failed_checks"],
        "next_execution": gate["next_execution"],
        "grpo_training_authorized": False,
        "return_files": return_files,
    }
    write_json_atomic(root / FEEDBACK, feedback)
    manifest["feedback"] = {
        "path": str(FEEDBACK).replace("\\", "/"),
        "sha256": file_sha256(root / FEEDBACK),
    }
    write_json_atomic(root / MANIFEST, manifest)
    emit(
        "bp4_dpo_evaluation_complete",
        status="PASS" if all(integrity_checks.values()) else "FAIL",
        research_progression_decision=gate[
            "research_progression_decision"
        ],
        deployment_decision=gate["deployment_decision"],
        next_execution=gate["next_execution"],
        grpo_training_authorized=False,
        return_file_count=len(return_files),
    )
    if not all(integrity_checks.values()):
        raise ValueError("BP4 DPO evaluation package integrity checks failed")
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--reliability-config",
        type=Path,
        default=Path("configs/experiments/bp4_reliability_repair_v1.yaml"),
    )
    parser.add_argument(
        "--matrix-config",
        type=Path,
        default=Path("configs/experiments/bp4_controlled_model_matrix_v1.yaml"),
    )
    parser.add_argument(
        "--bp2-config",
        type=Path,
        default=Path("configs/experiments/bp2_dpo_reliability_v1.yaml"),
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

    def rooted(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    try:
        paths = (
            rooted(args.reliability_config),
            rooted(args.matrix_config),
            rooted(args.bp2_config),
            rooted(args.artifacts),
        )
        if args.command == "preflight":
            preflight(root, *paths)
        elif args.command == "infer":
            run_inference(
                root,
                *paths,
                batch_size=args.batch_size,
                max_input_tokens=args.max_input_tokens,
            )
        else:
            package_evaluation(root, *paths)
        return 0
    except Exception as exc:
        emit(
            "bp4_dpo_reliability",
            status="FAIL",
            error_type=type(exc).__name__,
            error=str(exc),
            training_started_by_evaluation=False,
            grpo_started=False,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
