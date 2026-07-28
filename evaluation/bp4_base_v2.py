"""Run and package the BP4 Base v2 reference evaluation."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import yaml

from core.target_contract import (
    DATASET_VERSION,
    TARGET_CONTRACT_VERSION,
    build_messages,
    validate_keyword_payload,
)
from evaluation.bp2_evaluator import evaluate_with_slices, prepare_reference
from evaluation.keyword_evaluator import (
    file_sha256,
    read_jsonl,
    render_markdown,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from scripts.bp4_matrix import base_task, load_config, validate_dependencies
from scripts.build_bp1_assets import source_and_payload
from scripts.prepare_experiment_manifest import (
    _environment_lineage,
    _git_lineage,
    _source_lineage,
)


PACKET_VERSION = "bp4-base-evaluation-v1"
REPORT_VERSION = "bp4-base-v2-report-v1"


def emit(event: str, **payload: Any) -> None:
    print(
        json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


def prepare_inference_rows(
    rows: Sequence[Mapping[str, Any]], expected_split: str
) -> list[Dict[str, Any]]:
    prepared = []
    seen = set()
    for row_number, row in enumerate(rows, 1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(
                f"Invalid or duplicate {expected_split} sample_id at row {row_number}"
            )
        seen.add(sample_id)
        if row.get("dataset_version") != DATASET_VERSION:
            raise ValueError(f"Dataset version mismatch for {sample_id}")
        if row.get("target_contract_version") != TARGET_CONTRACT_VERSION:
            raise ValueError(f"Target contract mismatch for {sample_id}")
        if row.get("split") != expected_split:
            raise ValueError(f"Split mismatch for {sample_id}")
        source, payload = source_and_payload(row)
        if row.get("messages") != build_messages(source, payload):
            raise ValueError(f"Prompt/target contract drift for {sample_id}")
        prepared.append(
            {
                "sample_id": sample_id,
                "source": source,
                "messages": build_messages(source),
                "challenge_slices": row.get("challenge_slices") or [],
            }
        )
    return sorted(prepared, key=lambda item: item["sample_id"])


def parse_completion(
    sample_id: str, raw_response: str, source: str
) -> Dict[str, Any]:
    try:
        payload = json.loads(raw_response.strip())
    except json.JSONDecodeError as exc:
        return {
            "sample_id": sample_id,
            "status": "error",
            "data": None,
            "error_code": "exact_json_parse_failed",
            "validation_errors": [exc.msg],
            "raw_response": raw_response,
        }
    valid, errors = validate_keyword_payload(payload, source)
    if not valid:
        return {
            "sample_id": sample_id,
            "status": "error",
            "data": None,
            "error_code": "target_contract_failed",
            "validation_errors": errors,
            "raw_response": raw_response,
        }
    return {
        "sample_id": sample_id,
        "status": "success",
        "data": payload,
        "raw_response": raw_response,
    }


def finish_reason(
    generated_token_ids: Sequence[int],
    *,
    eos_token_id: int,
    max_new_tokens: int,
) -> str:
    return (
        "length"
        if len(generated_token_ids) >= max_new_tokens
        and eos_token_id not in generated_token_ids
        else "stop"
    )


def _load_task(
    root: Path, config_path: Path, task_path: Path
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    config = load_config(config_path)
    validate_dependencies(root, config)
    bp2_path = root / config["pipeline"]["config"]
    bp2 = yaml.safe_load(bp2_path.read_text(encoding="utf-8"))
    expected = base_task(config, bp2)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    if task != expected:
        raise ValueError("Base v2 task does not match the frozen BP4 config")
    return config, task


def _load_existing_predictions(
    path: Path, allowed_ids: set[str]
) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        return {}
    indexed = {}
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        if (
            not isinstance(sample_id, str)
            or sample_id not in allowed_ids
            or sample_id in indexed
        ):
            raise ValueError(f"Invalid resumable prediction file: {path}")
        indexed[sample_id] = row
    return indexed


def _run_split(
    *,
    model: Any,
    tokenizer: Any,
    rows: Sequence[Dict[str, Any]],
    prediction_path: Path,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    import torch

    allowed_ids = {row["sample_id"] for row in rows}
    completed = _load_existing_predictions(prediction_path, allowed_ids)
    pending = [row for row in rows if row["sample_id"] not in completed]
    started = time.perf_counter()
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True
            )
            for row in batch
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        ).to(model.device)
        batch_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        batch_seconds = time.perf_counter() - batch_started
        generated_only = generated[:, inputs["input_ids"].shape[1] :]
        responses = tokenizer.batch_decode(generated_only, skip_special_tokens=True)
        for row, response, attention_mask, token_ids in zip(
            batch, responses, inputs["attention_mask"], generated_only
        ):
            prediction = parse_completion(
                row["sample_id"], response, row["source"]
            )
            prediction.update(
                {
                    "input_tokens": int(attention_mask.sum().item()),
                    "output_tokens": len(
                        tokenizer.encode(response, add_special_tokens=False)
                    ),
                    "finish_reason": finish_reason(
                        token_ids.tolist(),
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=max_new_tokens,
                    ),
                    "request_seconds": batch_seconds / len(batch),
                    "target_contract_version": TARGET_CONTRACT_VERSION,
                }
            )
            completed[row["sample_id"]] = prediction
        write_jsonl_atomic(
            prediction_path,
            [completed[row["sample_id"]] for row in rows if row["sample_id"] in completed],
        )
        emit(
            "bp4_base_batch_complete",
            prediction_file=str(prediction_path),
            completed=len(completed),
            total=len(rows),
        )
    if set(completed) != allowed_ids:
        raise ValueError(f"Incomplete Base predictions: {prediction_path}")
    return {
        "rows": len(rows),
        "new_rows": len(pending),
        "seconds": time.perf_counter() - started,
        "prediction_sha256": file_sha256(prediction_path),
    }


def run_inference(
    root: Path,
    config_path: Path,
    task_path: Path,
    *,
    batch_size: int,
    max_input_tokens: int,
) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if batch_size < 1 or max_input_tokens < 1:
        raise ValueError("Batch size and max input tokens must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BP4 Base v2 inference")
    config, task = _load_task(root, config_path, task_path)
    dev_path = root / task["selection"]["path"]
    challenge_path = root / task["diagnostic"]["path"]
    dev_rows = prepare_inference_rows(read_jsonl(dev_path), "dev")
    challenge_rows = prepare_inference_rows(
        read_jsonl(challenge_path), "challenge"
    )
    if len(dev_rows) != task["selection"]["rows"]:
        raise ValueError("Base v2 dev row count mismatch")
    if len(challenge_rows) != task["diagnostic"]["rows"]:
        raise ValueError("Base v2 challenge row count mismatch")
    outputs = task["required_outputs"]
    output_dir = (root / outputs["base_packet"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": "bp4-base-inference-v1",
        "task_path": str(task_path.relative_to(root)).replace("\\", "/"),
        "task_sha256": file_sha256(task_path),
        "model_path": task["model_path"],
        "target_contract_version": task["target_contract_version"],
        "dev_data_sha256": file_sha256(dev_path),
        "challenge_data_sha256": file_sha256(challenge_path),
        "decoding": task["decoding"],
        "batch_size": batch_size,
        "max_input_tokens": max_input_tokens,
        "human_gold_used": False,
        "training_started": False,
    }
    manifest_path = output_dir / "inference_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("Existing Base inference manifest does not match")
    else:
        write_json_atomic(manifest_path, manifest)

    torch.manual_seed(task["decoding"]["seed"])
    torch.cuda.manual_seed_all(task["decoding"]["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        task["model_path"], trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        task["model_path"],
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    dev_result = _run_split(
        model=model,
        tokenizer=tokenizer,
        rows=dev_rows,
        prediction_path=root / outputs["dev_predictions"],
        batch_size=batch_size,
        max_input_tokens=max_input_tokens,
        max_new_tokens=task["decoding"]["max_new_tokens"],
    )
    challenge_result = _run_split(
        model=model,
        tokenizer=tokenizer,
        rows=challenge_rows,
        prediction_path=root / outputs["challenge_predictions"],
        batch_size=batch_size,
        max_input_tokens=max_input_tokens,
        max_new_tokens=task["decoding"]["max_new_tokens"],
    )
    runtime = {
        "report_version": "bp4-base-runtime-v1",
        "dev": dev_result,
        "challenge": challenge_result,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "training_started": False,
    }
    write_json_atomic(output_dir / "inference_runtime.json", runtime)
    emit(
        "bp4_base_inference_complete",
        status="PASS",
        dev_rows=len(dev_rows),
        challenge_rows=len(challenge_rows),
        training_started=False,
    )
    return runtime


def _metric_summary(
    report: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    max_new_tokens: int,
) -> Dict[str, Any]:
    resources = report["resource_metrics"]
    return {
        "micro_f1": report["micro"]["f1"],
        "macro_f1": report["macro"]["f1"],
        "schema_valid_rate": report["schema_valid_rate"],
        "hallucination_keyword_rate": report["hallucination"]["keyword_rate"],
        "mean_input_tokens": resources["mean_input_tokens"],
        "mean_output_tokens": resources["mean_output_tokens"],
        "mean_request_seconds": resources["mean_request_seconds"],
        "max_token_output_count": sum(
            row.get("finish_reason") == "length" for row in predictions
        ),
        "error_sample_count": report["error_sample_count"],
    }


def _validate_prediction_metadata(
    predictions: Sequence[Mapping[str, Any]]
) -> None:
    for row in predictions:
        if row.get("target_contract_version") != TARGET_CONTRACT_VERSION:
            raise ValueError(
                "Base predictions lack the frozen target contract version"
            )
        if row.get("finish_reason") not in {"stop", "length"}:
            raise ValueError("Base prediction has invalid finish reason")
        for name in ("input_tokens", "output_tokens", "request_seconds"):
            value = row.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0
            ):
                raise ValueError(f"Base prediction has invalid {name}")


def package_evaluation(
    root: Path, config_path: Path, task_path: Path
) -> Dict[str, Any]:
    config, task = _load_task(root, config_path, task_path)
    bp4_gate_path = root / config["outputs"]["gate"]
    bp4_gate = json.loads(bp4_gate_path.read_text(encoding="utf-8"))
    if (
        bp4_gate.get("decision") != "ACCEPT_BP4_MATRIX_ENGINEERING"
        or bp4_gate.get("status") != "PASS"
        or bp4_gate.get("training_started") is not False
    ):
        raise ValueError("BP4 engineering gate is not accepted")

    outputs = task["required_outputs"]
    dev_path = root / task["selection"]["path"]
    challenge_path = root / task["diagnostic"]["path"]
    dev_predictions_path = root / outputs["dev_predictions"]
    challenge_predictions_path = root / outputs["challenge_predictions"]
    dev_rows = read_jsonl(dev_path)
    challenge_rows = read_jsonl(challenge_path)
    prepare_inference_rows(dev_rows, "dev")
    prepare_inference_rows(challenge_rows, "challenge")
    dev_predictions = read_jsonl(dev_predictions_path)
    challenge_predictions = read_jsonl(challenge_predictions_path)
    _validate_prediction_metadata(dev_predictions + challenge_predictions)
    dev_reference = prepare_reference(dev_rows)
    challenge_reference = prepare_reference(challenge_rows)
    dev_report, dev_errors = evaluate_with_slices(
        dev_reference, dev_predictions, "B1_V2"
    )
    challenge_report, challenge_errors = evaluate_with_slices(
        challenge_reference, challenge_predictions, "B1_V2:challenge"
    )
    write_json_atomic(root / outputs["dev_evaluation"], dev_report)
    write_json_atomic(root / outputs["challenge_evaluation"], challenge_report)
    output_dir = (root / outputs["base_packet"]).parent
    write_jsonl_atomic(output_dir / "dev_reference.jsonl", dev_reference)
    write_jsonl_atomic(output_dir / "challenge_reference.jsonl", challenge_reference)
    write_jsonl_atomic(output_dir / "dev_errors.jsonl", dev_errors)
    write_jsonl_atomic(output_dir / "challenge_errors.jsonl", challenge_errors)
    write_text_atomic(
        output_dir / "dev_evaluation.md", render_markdown(dev_report)
    )
    write_text_atomic(
        output_dir / "challenge_evaluation.md", render_markdown(challenge_report)
    )

    max_new_tokens = task["decoding"]["max_new_tokens"]
    dev_metrics = _metric_summary(dev_report, dev_predictions, max_new_tokens)
    challenge_metrics = _metric_summary(
        challenge_report, challenge_predictions, max_new_tokens
    )
    packet = {
        "packet_version": PACKET_VERSION,
        "status": "PENDING",
        "model_id": "B1_V2",
        "model_path": task["model_path"],
        "target_contract_version": task["target_contract_version"],
        "dev_rows": len(dev_reference),
        "challenge_rows": len(challenge_reference),
        "selection_data_sha256": file_sha256(dev_path),
        "diagnostic_data_sha256": file_sha256(challenge_path),
        "dev_predictions_sha256": file_sha256(dev_predictions_path),
        "challenge_predictions_sha256": file_sha256(challenge_predictions_path),
        "dev_evaluation_sha256": file_sha256(root / outputs["dev_evaluation"]),
        "challenge_evaluation_sha256": file_sha256(
            root / outputs["challenge_evaluation"]
        ),
        "decoding": task["decoding"],
        "dev_metrics": dev_metrics,
        "challenge_metrics": challenge_metrics,
        "challenge_usage": "diagnostic_only",
        "human_gold_used": False,
        "training_started": False,
    }
    inference_manifest_path = output_dir / "inference_manifest.json"
    inference_manifest = json.loads(
        inference_manifest_path.read_text(encoding="utf-8")
    )
    runtime_path = output_dir / "inference_runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    checks = {
        "bp4_engineering_gate_accepted": True,
        "dev_data_hash_locked": (
            packet["selection_data_sha256"] == task["selection"]["sha256"]
        ),
        "challenge_data_hash_locked": (
            packet["diagnostic_data_sha256"] == task["diagnostic"]["sha256"]
        ),
        "dev_coverage_exact_331": len(dev_reference) == 331
        and len(dev_predictions) == 331,
        "challenge_coverage_exact_140": len(challenge_reference) == 140
        and len(challenge_predictions) == 140,
        "target_contract_v2": (
            packet["target_contract_version"] == TARGET_CONTRACT_VERSION
        ),
        "decoding_frozen": packet["decoding"] == task["decoding"],
        "inference_manifest_bound": (
            inference_manifest.get("task_sha256") == file_sha256(task_path)
            and inference_manifest.get("model_path") == task["model_path"]
            and inference_manifest.get("decoding") == task["decoding"]
            and inference_manifest.get("human_gold_used") is False
            and inference_manifest.get("training_started") is False
        ),
        "challenge_diagnostic_only": (
            packet["challenge_usage"] == "diagnostic_only"
        ),
        "human_gold_not_used": packet["human_gold_used"] is False,
        "training_not_started": packet["training_started"] is False,
        "runtime_reports_no_training": runtime.get("training_started") is False,
    }
    passed = all(checks.values())
    packet["status"] = "PASS" if passed else "FAIL"
    packet["runtime"] = {
        "dev_seconds": runtime["dev"]["seconds"],
        "challenge_seconds": runtime["challenge"]["seconds"],
        "peak_memory_gib": runtime["peak_memory_gib"],
    }
    write_json_atomic(root / outputs["base_packet"], packet)
    report_outputs = {
        "base_packet": root / outputs["base_packet"],
        "dev_predictions": dev_predictions_path,
        "challenge_predictions": challenge_predictions_path,
        "dev_evaluation": root / outputs["dev_evaluation"],
        "challenge_evaluation": root / outputs["challenge_evaluation"],
    }
    manifest = {
        "manifest_version": REPORT_VERSION,
        "config": {
            "path": str(config_path.relative_to(root)).replace("\\", "/"),
            "sha256": file_sha256(config_path),
        },
        "task": {
            "path": str(task_path.relative_to(root)).replace("\\", "/"),
            "sha256": file_sha256(task_path),
        },
        "bp4_engineering_gate": {
            "path": str(bp4_gate_path.relative_to(root)).replace("\\", "/"),
            "sha256": file_sha256(bp4_gate_path),
        },
        "inference": {
            "manifest": inference_manifest,
            "manifest_sha256": file_sha256(inference_manifest_path),
            "runtime": runtime,
            "runtime_sha256": file_sha256(runtime_path),
        },
        "outputs": {
            name: {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": file_sha256(path),
            }
            for name, path in report_outputs.items()
        },
        "checks": checks,
        "source": _source_lineage(root),
        "git": _git_lineage(root),
        "environment": _environment_lineage(),
        "training_started": False,
    }
    manifest_path = output_dir / "base_manifest.json"
    write_json_atomic(manifest_path, manifest)
    gate = {
        "report_version": REPORT_VERSION,
        "status": "PASS" if passed else "FAIL",
        "decision": (
            "ACCEPT_BP4_BASE_V2" if passed else "REJECT_BP4_BASE_V2"
        ),
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "manifest_sha256": file_sha256(manifest_path),
        "base_packet_sha256": file_sha256(root / outputs["base_packet"]),
        "next_execution": (
            "BP4_SFT_AUTHORIZATION_REVIEW" if passed else "STOP_AND_DIAGNOSE_BASE"
        ),
        "authorization": (
            "inactive_sft_candidate_may_be_generated; training_still_blocked"
        ),
        "training_started": False,
    }
    gate_path = output_dir / "base_gate.json"
    write_json_atomic(gate_path, gate)
    feedback = {
        "packet_version": "bp4-base-feedback-v1",
        "decision": gate["decision"],
        "failed_checks": gate["failed_checks"],
        "manifest_sha256": gate["manifest_sha256"],
        "gate_sha256": file_sha256(gate_path),
        "base_packet_sha256": gate["base_packet_sha256"],
        "next_execution": gate["next_execution"],
        "authorization": gate["authorization"],
        "training_started": False,
    }
    write_json_atomic(output_dir / "base_feedback.json", feedback)
    emit(
        "bp4_base_gate_complete",
        status=gate["status"],
        decision=gate["decision"],
        gate_file=str(gate_path.relative_to(root)).replace("\\", "/"),
        gate_sha256=file_sha256(gate_path),
        base_packet_sha256=gate["base_packet_sha256"],
        training_started=False,
    )
    if not passed:
        raise ValueError(f"BP4 Base v2 gate failed: {gate['failed_checks']}")
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/bp4_controlled_model_matrix_v1.yaml"),
    )
    parser.add_argument(
        "--task",
        type=Path,
        default=Path(
            "reports/bp4_controlled_matrix/base_v2_evaluation_task.json"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    infer = subparsers.add_parser("infer")
    infer.add_argument("--batch-size", type=int, default=4)
    infer.add_argument("--max-input-tokens", type=int, default=4096)
    subparsers.add_parser("package")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    task_path = args.task if args.task.is_absolute() else root / args.task
    try:
        if args.command == "infer":
            run_inference(
                root,
                config_path,
                task_path,
                batch_size=args.batch_size,
                max_input_tokens=args.max_input_tokens,
            )
        else:
            package_evaluation(root, config_path, task_path)
        return 0
    except Exception as exc:
        emit(
            "bp4_base_v2",
            status="FAIL",
            error_type=type(exc).__name__,
            error=str(exc),
            training_started=False,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
