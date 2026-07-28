"""Frozen final-checkpoint evaluation for E4 GRPO_FROM_DPO."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from core.target_contract import TARGET_CONTRACT_VERSION
from evaluation.bp2_evaluator import evaluate_with_slices, prepare_reference
from evaluation.bp4_base_v2 import (
    _run_split,
    _validate_prediction_metadata,
    prepare_inference_rows,
)
from evaluation.bp4_contract_telemetry import contract_aware_telemetry
from evaluation.bp4_stage_common import metric_summary, paired_bootstrap, sample_f1
from evaluation.keyword_evaluator import file_sha256, read_jsonl, write_json_atomic, write_jsonl_atomic
from scripts.bp2_pipeline import load_config as load_bp2_config, read_authorization
from scripts.prepare_experiment_manifest import _environment_lineage, _git_lineage, _source_lineage


OUTPUT_DIR = Path("reports/bp4_e4_grpo_from_dpo/evaluation")
DEV_PREDICTIONS = OUTPUT_DIR / "dev_predictions.jsonl"
CHALLENGE_PREDICTIONS = OUTPUT_DIR / "challenge_predictions.jsonl"
PAIR_PACKET = OUTPUT_DIR / "grpo_paired_evaluation_packet.json"
GATE = OUTPUT_DIR / "grpo_two_level_gate.json"
MANIFEST = OUTPUT_DIR / "grpo_evaluation_manifest.json"
FEEDBACK = OUTPUT_DIR / "grpo_evaluation_feedback.json"


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True), flush=True)


def load_config(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "structalign-bp4-e4-evaluation-v1":
        raise ValueError("Invalid E4 evaluation config")
    return value


def _rooted(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def select_last_checkpoint(root: Path, artifact_path: Path) -> Path:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    latest = artifact.get("latest_checkpoint")
    if not isinstance(latest, dict) or not isinstance(latest.get("checkpoint"), str):
        raise ValueError("E4 artifacts do not identify latest_checkpoint")
    checkpoint = _rooted(root, latest["checkpoint"])
    if not checkpoint.is_dir() or not (checkpoint / "adapter_config.json").is_file():
        raise ValueError("E4 final checkpoint adapter artifacts are missing")
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file() or json.loads(state_path.read_text(encoding="utf-8")).get("global_step", 0) <= 0:
        raise ValueError("E4 final checkpoint lacks a completed trainer state")
    return checkpoint


def _metric_summary(
    report: Mapping[str, Any],
    predictions: list[Mapping[str, Any]],
    telemetry: Mapping[str, Any],
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Accept both deployed BP4 helper signatures while preserving E4 metrics."""
    try:
        summary = metric_summary(report, predictions, telemetry, max_new_tokens)
    except TypeError:
        summary = metric_summary(report, predictions, telemetry)
    summary.setdefault("max_token_output_count", sum(
        row.get("output_tokens", 0) >= max_new_tokens for row in predictions
    ))
    summary.setdefault(
        "hallucination_metric_version", telemetry["telemetry_version"]
    )
    return summary


def preflight(root: Path, evaluation_path: Path, bp2_path: Path) -> Dict[str, Any]:
    evaluation = load_config(evaluation_path)
    bp2 = load_bp2_config(bp2_path)
    authorization, blockers = read_authorization(bp2, "grpo", root, config_sha256=file_sha256(bp2_path))
    if blockers:
        raise ValueError(f"E4 authorization blockers: {blockers}")
    inputs = evaluation["inputs"]
    artifact_path = _rooted(root, inputs["training_artifacts"])
    dpo_packet_path = _rooted(root, inputs["dpo_paired_packet"])
    dpo_predictions = _rooted(root, inputs["dpo_dev_predictions"])
    packet = json.loads(dpo_packet_path.read_text(encoding="utf-8"))
    checkpoint = select_last_checkpoint(root, artifact_path)
    checks = {
        "authorization_active": authorization.get("activation_status") == "ACTIVE_APPROVED",
        "scope_is_e4_only": authorization.get("approved_scope") == "E4_GRPO_FROM_DPO_ONLY",
        "matrix_id_locked": authorization.get("matrix_id") == "E4_GRPO_FROM_DPO",
        "direct_upstream_is_dpo": Path(authorization["model_path"]).resolve() == Path(packet["candidate_checkpoint"]).resolve(),
        "dpo_predictions_bound": file_sha256(dpo_predictions) == packet["candidate_predictions_sha256"],
        "final_checkpoint_has_steps": json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8")).get("global_step", 0) > 0,
        "human_gold_forbidden": evaluation["evaluation"]["forbidden_selection"] == "frozen_human_gold",
    }
    if not all(checks.values()):
        raise ValueError(f"E4 evaluation preflight failed: {[name for name, value in checks.items() if not value]}")
    report = {
        "report_version": "bp4-e4-evaluation-preflight-v1", "status": "PASS", "checks": checks,
        "candidate_checkpoint": str(checkpoint), "direct_upstream_checkpoint": authorization["model_path"],
        "authorization_sha256": file_sha256(root / bp2["stages"]["grpo"]["authorization_file"]),
        "training_artifacts_sha256": file_sha256(artifact_path), "human_gold_used": False,
        "training_started_by_evaluation": False,
    }
    write_json_atomic(root / OUTPUT_DIR / "preflight.json", report)
    emit("bp4_e4_grpo_evaluation_preflight", **report)
    return report


def run_inference(root: Path, evaluation_path: Path, bp2_path: Path, *, batch_size: int, max_input_tokens: int) -> Dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for E4 inference")
    preflight_report = preflight(root, evaluation_path, bp2_path)
    evaluation = load_config(evaluation_path)
    bp2 = load_bp2_config(bp2_path)
    protocol = evaluation["evaluation"]
    dev_path, challenge_path = (_rooted(root, protocol["selection"]["path"]), _rooted(root, protocol["diagnostic"]["path"]))
    dev_rows, challenge_rows = (prepare_inference_rows(read_jsonl(dev_path), "dev"), prepare_inference_rows(read_jsonl(challenge_path), "challenge"))
    if len(dev_rows) != protocol["selection"]["rows"] or len(challenge_rows) != protocol["diagnostic"]["rows"]:
        raise ValueError("E4 evaluation split row count mismatch")
    output_dir = root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": "bp4-e4-inference-v1", "candidate_checkpoint": preflight_report["candidate_checkpoint"],
        "direct_upstream_checkpoint": preflight_report["direct_upstream_checkpoint"],
        "selection_data_sha256": file_sha256(dev_path), "diagnostic_data_sha256": file_sha256(challenge_path),
        "decoding": protocol["decoding"], "batch_size": batch_size, "max_input_tokens": max_input_tokens,
        "checkpoint_selection": "last_completed_checkpoint", "human_gold_used": False, "training_started_by_evaluation": False,
    }
    write_json_atomic(output_dir / "inference_manifest.json", manifest)
    torch.manual_seed(protocol["decoding"]["seed"]); torch.cuda.manual_seed_all(protocol["decoding"]["seed"])
    tokenizer = AutoTokenizer.from_pretrained(bp2["model"]["base_path"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(bp2["model"]["base_path"], dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, preflight_report["candidate_checkpoint"]).cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    dev = _run_split(model=model, tokenizer=tokenizer, rows=dev_rows, prediction_path=root / DEV_PREDICTIONS, batch_size=batch_size, max_input_tokens=max_input_tokens, max_new_tokens=protocol["decoding"]["max_new_tokens"])
    challenge = _run_split(model=model, tokenizer=tokenizer, rows=challenge_rows, prediction_path=root / CHALLENGE_PREDICTIONS, batch_size=batch_size, max_input_tokens=max_input_tokens, max_new_tokens=protocol["decoding"]["max_new_tokens"])
    runtime = {"report_version": "bp4-e4-runtime-v1", "candidate_checkpoint": preflight_report["candidate_checkpoint"], "dev": dev, "challenge": challenge, "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024**3), "training_started_by_evaluation": False}
    write_json_atomic(output_dir / "inference_runtime.json", runtime)
    emit("bp4_e4_grpo_inference_complete", status="PASS", candidate_checkpoint=preflight_report["candidate_checkpoint"], dev_rows=len(dev_rows), challenge_rows=len(challenge_rows), training_started_by_evaluation=False)
    return runtime


def research_checks(packet: Mapping[str, Any], gates: Mapping[str, Any]) -> Dict[str, bool]:
    baseline, candidate, paired = packet["baseline"], packet["candidate"], packet["paired"]
    return {
        "complete_dev_coverage": packet["sample_count"] == 331,
        "macro_f1_delta_min": paired["macro_f1_delta_candidate_minus_baseline"] >= gates["macro_f1_delta_min"],
        "macro_f1_ci_low_gt": paired["macro_f1_bootstrap_95_ci"][0] > gates["macro_f1_ci_low_gt"],
        "micro_f1_delta_min": candidate["micro_f1"] - baseline["micro_f1"] >= gates["micro_f1_delta_min"],
        "schema_valid_rate_min": candidate["schema_valid_rate"] >= gates["schema_valid_rate_min"],
        "schema_valid_rate_delta_min": candidate["schema_valid_rate"] - baseline["schema_valid_rate"] >= gates["schema_valid_rate_delta_min"],
        "hallucination_keyword_rate_max": candidate["hallucination_keyword_rate"] <= gates["hallucination_keyword_rate_max"],
        "hallucination_keyword_rate_delta_max": candidate["hallucination_keyword_rate"] - baseline["hallucination_keyword_rate"] <= gates["hallucination_keyword_rate_delta_max"],
        "mean_output_tokens_ratio_max": candidate["mean_output_tokens"] / max(baseline["mean_output_tokens"], 1e-12) <= gates["mean_output_tokens_ratio_max"],
        "max_token_output_count_delta_max": candidate["max_token_output_count"] - baseline["max_token_output_count"] <= gates["max_token_output_count_delta_max"],
        "human_gold_not_used": packet["human_gold_used"] is False,
        "challenge_not_used_for_selection": packet["challenge_usage"] == "diagnostic_only",
    }


def package_evaluation(root: Path, evaluation_path: Path, bp2_path: Path) -> Dict[str, Any]:
    preflight_report = preflight(root, evaluation_path, bp2_path)
    config = load_config(evaluation_path); protocol, inputs = config["evaluation"], config["inputs"]
    dpo_packet = json.loads(_rooted(root, inputs["dpo_paired_packet"]).read_text(encoding="utf-8"))
    base_report = json.loads(_rooted(root, inputs["base_dev_evaluation"]).read_text(encoding="utf-8"))
    dev_reference, challenge_reference = prepare_reference(read_jsonl(_rooted(root, protocol["selection"]["path"]))), prepare_reference(read_jsonl(_rooted(root, protocol["diagnostic"]["path"])))
    upstream, candidate, challenge = read_jsonl(_rooted(root, inputs["dpo_dev_predictions"])), read_jsonl(root / DEV_PREDICTIONS), read_jsonl(root / CHALLENGE_PREDICTIONS)
    _validate_prediction_metadata(upstream + candidate + challenge)
    if len(upstream) != 331 or len(candidate) != 331 or len(challenge) != 140:
        raise ValueError("E4 evaluation coverage must be 331/331/140")
    upstream_report, _ = evaluate_with_slices(dev_reference, upstream, "E2_DPO_RELIABILITY")
    candidate_report, dev_errors = evaluate_with_slices(dev_reference, candidate, "E4_GRPO_FROM_DPO")
    challenge_report, challenge_errors = evaluate_with_slices(challenge_reference, challenge, "E4_GRPO_FROM_DPO:challenge")
    source_dev = {row["sample_id"]: row["normalized_text"] for row in dev_reference}; source_challenge = {row["sample_id"]: row["normalized_text"] for row in challenge_reference}
    upstream_telemetry, candidate_telemetry, challenge_telemetry = contract_aware_telemetry(upstream, source_dev), contract_aware_telemetry(candidate, source_dev), contract_aware_telemetry(challenge, source_challenge)
    max_tokens = protocol["decoding"]["max_new_tokens"]
    baseline_metrics, candidate_metrics = _metric_summary(upstream_report, upstream, upstream_telemetry, max_tokens), _metric_summary(candidate_report, candidate, candidate_telemetry, max_tokens)
    paired = paired_bootstrap(sample_f1(dev_reference, upstream), sample_f1(dev_reference, candidate), iterations=protocol["bootstrap_iterations"], seed=protocol["bootstrap_seed"])
    checkpoint = preflight_report["candidate_checkpoint"]
    packet = {"packet_version": "bp4-e4-paired-evaluation-v1", "selection_split": "teacher_dev", "sample_count": 331, "human_gold_used": False, "challenge_usage": "diagnostic_only", "target_contract_version": TARGET_CONTRACT_VERSION, "bootstrap_iterations": protocol["bootstrap_iterations"], "bootstrap_seed": protocol["bootstrap_seed"], "selection_data_sha256": file_sha256(_rooted(root, protocol["selection"]["path"])), "baseline_predictions_sha256": file_sha256(_rooted(root, inputs["dpo_dev_predictions"])), "candidate_predictions_sha256": file_sha256(root / DEV_PREDICTIONS), "baseline_model_id": "E2_DPO_RELIABILITY", "candidate_model_id": "E4_GRPO_FROM_DPO", "baseline_checkpoint": dpo_packet["candidate_checkpoint"], "candidate_checkpoint": checkpoint, "baseline_stage": "dpo", "baseline": baseline_metrics, "candidate": candidate_metrics, "paired": paired, "challenge": {"sample_count": 140, "usage": "diagnostic_only", "metrics": _metric_summary(challenge_report, challenge, challenge_telemetry, max_tokens)}}
    output = root / OUTPUT_DIR; output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(root / PAIR_PACKET, packet); write_json_atomic(output / "dev_evaluation.json", candidate_report); write_json_atomic(output / "challenge_evaluation.json", challenge_report); write_jsonl_atomic(output / "dev_errors.jsonl", dev_errors); write_jsonl_atomic(output / "challenge_errors.jsonl", challenge_errors); write_json_atomic(output / "upstream_contract_telemetry.json", upstream_telemetry); write_json_atomic(output / "candidate_contract_telemetry.json", {"dev": candidate_telemetry, "challenge": challenge_telemetry, "challenge_usage": "diagnostic_only"})
    research = research_checks(packet, config["research_gate"]); deployment = {"schema_valid_rate_min": candidate_metrics["schema_valid_rate"] >= config["deployment_gate"]["schema_valid_rate_min"], "hallucination_keyword_rate_max": candidate_metrics["hallucination_keyword_rate"] <= config["deployment_gate"]["hallucination_keyword_rate_max"], "mean_output_tokens_vs_base_ratio_max": candidate_metrics["mean_output_tokens"] / max(base_report["resource_metrics"]["mean_output_tokens"], 1e-12) <= config["deployment_gate"]["mean_output_tokens_vs_base_ratio_max"], "max_token_output_count": candidate_metrics["max_token_output_count"] == config["deployment_gate"]["max_token_output_count"], "research_progression_passed": all(research.values()), "human_gold_not_used": True, "challenge_not_used_for_selection": True}
    research_passed, deployment_passed = all(research.values()), all(deployment.values())
    gate = {"report_version": "bp4-e4-two-level-gate-v1", "status": "PASS", "stage": "grpo", "candidate_checkpoint": checkpoint, "direct_upstream_checkpoint": dpo_packet["candidate_checkpoint"], "research_progression_decision": "ACCEPT_E4_GRPO_RESEARCH_PROGRESSION" if research_passed else "REJECT_E4_GRPO_RESEARCH_PROGRESSION", "research_progression_checks": research, "research_progression_failed_checks": [name for name, value in research.items() if not value], "deployment_decision": "ACCEPT_E4_GRPO_DEPLOYMENT" if deployment_passed else "REJECT_E4_GRPO_DEPLOYMENT", "deployment_checks": deployment, "deployment_failed_checks": [name for name, value in deployment.items() if not value], "paired_packet_sha256": file_sha256(root / PAIR_PACKET), "human_gold_used": False, "deployment_authorized": False, "next_execution": "HUMAN_REVIEW_BP5_SERVING_CANDIDATE" if deployment_passed else "E4_EVALUATION_COMPLETE_HUMAN_REVIEW_REQUIRED"}
    write_json_atomic(root / GATE, gate)
    outputs = {"preflight": root / OUTPUT_DIR / "preflight.json", "inference_manifest": root / OUTPUT_DIR / "inference_manifest.json", "inference_runtime": root / OUTPUT_DIR / "inference_runtime.json", "dev_predictions": root / DEV_PREDICTIONS, "challenge_predictions": root / CHALLENGE_PREDICTIONS, "dev_evaluation": output / "dev_evaluation.json", "challenge_evaluation": output / "challenge_evaluation.json", "paired_packet": root / PAIR_PACKET, "two_level_gate": root / GATE}
    manifest = {"manifest_version": "bp4-e4-evaluation-package-v1", "evaluation_config_sha256": file_sha256(evaluation_path), "bp2_config_sha256": file_sha256(bp2_path), "authorization_sha256": preflight_report["authorization_sha256"], "candidate_checkpoint": checkpoint, "checkpoint_selection": "last_completed_checkpoint", "outputs": {name: {"path": str(path.relative_to(root)).replace("\\\\", "/"), "sha256": file_sha256(path)} for name, path in outputs.items()}, "checks": {"preflight_passed": True, "direct_upstream_is_dpo": True, "dev_coverage_exact_331": True, "challenge_coverage_exact_140": True, "human_gold_not_used": True, "challenge_diagnostic_only": True, "training_not_restarted": True}, "source": _source_lineage(root), "git": _git_lineage(root), "environment": _environment_lineage(), "human_gold_used": False, "training_started_by_evaluation": False}
    write_json_atomic(root / MANIFEST, manifest)
    feedback = {"packet_version": "bp4-e4-evaluation-feedback-v1", "research_progression_decision": gate["research_progression_decision"], "deployment_decision": gate["deployment_decision"], "research_progression_failed_checks": gate["research_progression_failed_checks"], "deployment_failed_checks": gate["deployment_failed_checks"], "next_execution": gate["next_execution"], "deployment_authorized": False, "return_files": ["new_plan/logs/bp4_e4_grpo_from_dpo_evaluation.log", *[str(path.relative_to(root)).replace("\\\\", "/") for path in outputs.values()], str(MANIFEST), str(FEEDBACK)]}
    write_json_atomic(root / FEEDBACK, feedback)
    emit("bp4_e4_grpo_evaluation_complete", status="PASS", research_progression_decision=gate["research_progression_decision"], deployment_decision=gate["deployment_decision"], deployment_authorized=False)
    return gate


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(".")); parser.add_argument("--evaluation-config", type=Path, default=Path("configs/experiments/bp4_e4_grpo_from_dpo_evaluation_v1.yaml")); parser.add_argument("--bp2-config", type=Path, default=Path("configs/experiments/bp2_e4_grpo_from_dpo_v1.yaml")); commands = parser.add_subparsers(dest="command", required=True); commands.add_parser("preflight"); infer = commands.add_parser("infer"); infer.add_argument("--batch-size", type=int, default=4); infer.add_argument("--max-input-tokens", type=int, default=4096); commands.add_parser("package"); args = parser.parse_args(); root = args.root.resolve(); evaluation_path = _rooted(root, str(args.evaluation_config)); bp2_path = _rooted(root, str(args.bp2_config))
    try:
        if args.command == "preflight": preflight(root, evaluation_path, bp2_path)
        elif args.command == "infer": run_inference(root, evaluation_path, bp2_path, batch_size=args.batch_size, max_input_tokens=args.max_input_tokens)
        else: package_evaluation(root, evaluation_path, bp2_path)
        return 0
    except Exception as exc:
        emit("bp4_e4_grpo_evaluation", status="FAIL", error_type=type(exc).__name__, error=str(exc), training_started_by_evaluation=False)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
