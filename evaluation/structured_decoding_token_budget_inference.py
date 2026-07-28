"""Two-arm replay for isolating the structured-output max-token budget."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict

from evaluation.baseline_inference import DEV_SAMPLE_IDS_SHA256, emit, prepare_dev_gold
from evaluation.keyword_evaluator import (
    file_sha256,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from evaluation.structured_decoding_inference import (
    MAX_INPUT_TOKENS,
    PROMPT_VARIANT,
    REFERENCE_SHA256,
    run_arm,
)


REPLAY_VARIANT = "b1_vllm_json_schema_512_replay_v1"
CANDIDATE_VARIANT = "b1_vllm_json_schema_768_v1"
EXPERIMENT_VERSION = "p1-structured-decoding-token-budget-v1"
ORIGINAL_CONTROL_SHA256 = "f232afc60205ccbb6e76453d9adffc8fbc3b3439e3cd019ea801c13c3c81b438"
ORIGINAL_CONSTRAINED_SHA256 = "dcaa0961c37397a738b1fcb4a985e19bcb443fe5de2c3af7f2ff01cd755a40e7"


def run(args: argparse.Namespace) -> int:
    from openai import OpenAI

    if args.workers != 4:
        raise ValueError("Token-budget experiment requires frozen workers=4")
    prior_paths = {
        "reference": args.prior_dir / "dev_teacher_reference.jsonl",
        "control": args.prior_dir / "b1_vllm_unconstrained_v1_predictions.jsonl",
        "constrained": args.prior_dir / "b1_vllm_json_schema_v1_predictions.jsonl",
    }
    expected_hashes = {
        "reference": REFERENCE_SHA256,
        "control": ORIGINAL_CONTROL_SHA256,
        "constrained": ORIGINAL_CONSTRAINED_SHA256,
    }
    for name, path in prior_paths.items():
        if file_sha256(path) != expected_hashes[name]:
            raise ValueError(f"Prior {name} artifact hash mismatch")

    reference = prepare_dev_gold(read_jsonl(args.dev_file))
    sample_ids_hash = hashlib.sha256(
        "\n".join(row["sample_id"] for row in reference).encode("utf-8")
    ).hexdigest()
    if len(reference) != 331 or sample_ids_hash != DEV_SAMPLE_IDS_SHA256:
        raise ValueError("Teacher-dev split contract mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_file = args.output_dir / "dev_teacher_reference.jsonl"
    write_jsonl_atomic(reference_file, reference)
    if file_sha256(reference_file) != REFERENCE_SHA256:
        raise ValueError("Generated teacher-dev reference hash mismatch")

    client = OpenAI(
        base_url=args.base_url.rstrip("/") + "/v1",
        api_key=args.api_key,
        timeout=args.timeout_seconds,
        max_retries=0,
    )
    available_models = [item.id for item in client.models.list().data]
    if args.model not in available_models:
        raise ValueError(f"Served model {args.model!r} not found; available={available_models}")

    arm_specs = (
        (REPLAY_VARIANT, 512),
        (CANDIDATE_VARIANT, 768),
    )
    files: Dict[str, Dict[str, str]] = {}
    arm_seconds: Dict[str, float] = {}
    for variant, max_tokens in arm_specs:
        started = time.perf_counter()
        predictions = run_arm(
            client=client,
            rows=reference,
            variant=variant,
            model=args.model,
            max_tokens=max_tokens,
            seed=42,
            workers=args.workers,
            constrained=True,
        )
        arm_seconds[variant] = time.perf_counter() - started
        prediction_file = args.output_dir / f"{variant}_predictions.jsonl"
        write_jsonl_atomic(prediction_file, predictions)
        files[variant] = {
            "path": str(prediction_file),
            "sha256": file_sha256(prediction_file),
        }

    manifest: Dict[str, Any] = {
        "experiment_version": EXPERIMENT_VERSION,
        "label": "teacher_dev_structured_decoding_token_budget_not_human_gold_test",
        "dev_file": str(args.dev_file),
        "dev_sha256": file_sha256(args.dev_file),
        "reference_sha256": file_sha256(reference_file),
        "prior_dir": str(args.prior_dir),
        "prior_hashes": expected_hashes,
        "base_url": args.base_url,
        "served_model": args.model,
        "base_prompt_variant": PROMPT_VARIANT,
        "response_format": "same_json_schema_both_arms",
        "temperature": 0,
        "seed": 42,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "workers": args.workers,
        "timeout_seconds": args.timeout_seconds,
        "arms": files,
        "arm_max_tokens": {
            REPLAY_VARIANT: 512,
            CANDIDATE_VARIANT: 768,
        },
        "arm_seconds": arm_seconds,
        "single_variable": "max_tokens_512_to_768",
    }
    manifest_file = args.output_dir / "run_manifest.json"
    write_json_atomic(manifest_file, manifest)
    emit(
        "p1_structured_decoding_token_budget_inference_complete",
        status="PASS",
        manifest_file=str(manifest_file),
        manifest_sha256=file_sha256(manifest_file),
        prediction_files=files,
        arm_seconds=arm_seconds,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="qwen2.5-7b-instruct")
    parser.add_argument(
        "--dev-file", type=Path,
        default=Path("data/canonical/keyword_v1/splits/grpo_dev.jsonl"),
    )
    parser.add_argument(
        "--prior-dir", type=Path,
        default=Path("reports/baselines/qwen2_5_7b_dev/v5_structured_decoding"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("reports/baselines/qwen2_5_7b_dev/v6_structured_decoding_768"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        emit(
            "p1_structured_decoding_token_budget_inference_complete",
            status="FAIL",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
