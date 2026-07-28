"""Two-arm vLLM inference for isolating JSON-Schema constrained decoding."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence

from evaluation.baseline_inference import (
    DEV_SAMPLE_IDS_SHA256,
    build_prompt,
    emit,
    parse_completion,
    prepare_dev_gold,
)
from evaluation.keyword_evaluator import (
    file_sha256,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from examples.keyword_extraction.schema import create_keyword_schema


REFERENCE_SHA256 = "2c4cd0e5eaf14dc4ade0ac857e70d5669d9ea4746608c46c6bf3df6df3d83dfe"
CONTROL_VARIANT = "b1_vllm_unconstrained_v1"
CONSTRAINED_VARIANT = "b1_vllm_json_schema_v1"
PROMPT_VARIANT = "b1_structured_v3"
EXPERIMENT_VERSION = "p1-structured-decoding-v1"
MAX_INPUT_TOKENS = 4096


def keyword_response_format() -> Dict[str, Any]:
    """Return the response_format payload accepted by vLLM's OpenAI API."""
    schema = create_keyword_schema().to_json_schema()
    schema["properties"].pop("category", None)
    schema["required"] = ["keywords"]
    schema["additionalProperties"] = False
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "keyword_extraction",
            "schema": schema,
        },
    }


def build_request(
    source_text: str,
    model: str,
    max_tokens: int,
    seed: int,
    constrained: bool,
) -> Dict[str, Any]:
    system_prompt, user_prompt = build_prompt(PROMPT_VARIANT, source_text)
    request: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    if constrained:
        request["response_format"] = keyword_response_format()
    return request


def _predict_one(
    client: Any,
    row: Dict[str, Any],
    variant: str,
    model: str,
    max_tokens: int,
    seed: int,
    constrained: bool,
) -> Dict[str, Any]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        **build_request(row["normalized_text"], model, max_tokens, seed, constrained)
    )
    elapsed = time.perf_counter() - started
    if len(response.choices) != 1:
        raise ValueError(f"Unexpected choice count for {row['sample_id']}: {len(response.choices)}")
    choice = response.choices[0]
    raw_response = choice.message.content or ""
    prediction = parse_completion(row["sample_id"], variant, raw_response)
    usage = response.usage
    if usage is None or usage.prompt_tokens is None or usage.completion_tokens is None:
        raise ValueError(f"Missing token usage for {row['sample_id']}")
    if usage.prompt_tokens > MAX_INPUT_TOKENS:
        raise ValueError(
            f"Prompt exceeds frozen {MAX_INPUT_TOKENS}-token budget for {row['sample_id']}: "
            f"{usage.prompt_tokens}"
        )
    prediction.update({
        "input_tokens": int(usage.prompt_tokens),
        "output_tokens": int(usage.completion_tokens),
        "finish_reason": choice.finish_reason,
        "request_seconds": elapsed,
        "experiment_version": EXPERIMENT_VERSION,
        "base_prompt_variant": PROMPT_VARIANT,
        "structured_output": constrained,
    })
    return prediction


def run_arm(
    client: Any,
    rows: Sequence[Dict[str, Any]],
    variant: str,
    model: str,
    max_tokens: int,
    seed: int,
    workers: int,
    constrained: bool,
) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _predict_one, client, row, variant, model, max_tokens, seed, constrained
            ): row["sample_id"]
            for row in rows
        }
        for completed, future in enumerate(as_completed(futures), 1):
            prediction = future.result()
            predictions.append(prediction)
            if completed % 25 == 0 or completed == len(futures):
                emit(
                    "p1_structured_decoding_progress",
                    variant=variant,
                    completed=completed,
                    total=len(futures),
                )
    return sorted(predictions, key=lambda item: item["sample_id"])


def run(args: argparse.Namespace) -> int:
    from openai import OpenAI

    if args.workers < 1 or args.max_tokens < 1:
        raise ValueError("workers and max-tokens must be positive")
    dev_rows = read_jsonl(args.dev_file)
    reference = prepare_dev_gold(dev_rows)
    if len(reference) != 331:
        raise ValueError(f"Unexpected teacher-dev row count: {len(reference)}")
    sample_ids_hash = hashlib.sha256(
        "\n".join(row["sample_id"] for row in reference).encode("utf-8")
    ).hexdigest()
    if sample_ids_hash != DEV_SAMPLE_IDS_SHA256:
        raise ValueError("Teacher-dev sample-id hash mismatch")

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
    model_list = client.models.list()
    available_models = [item.id for item in model_list.data]
    if args.model not in available_models:
        raise ValueError(f"Served model {args.model!r} not found; available={available_models}")

    arm_specs = (
        (CONTROL_VARIANT, False),
        (CONSTRAINED_VARIANT, True),
    )
    files = {}
    arm_seconds = {}
    for variant, constrained in arm_specs:
        started = time.perf_counter()
        predictions = run_arm(
            client, reference, variant, args.model, args.max_tokens,
            args.seed, args.workers, constrained,
        )
        arm_seconds[variant] = time.perf_counter() - started
        prediction_file = args.output_dir / f"{variant}_predictions.jsonl"
        write_jsonl_atomic(prediction_file, predictions)
        files[variant] = {
            "path": str(prediction_file),
            "sha256": file_sha256(prediction_file),
        }

    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "label": "teacher_dev_structured_decoding_not_human_gold_test",
        "dev_file": str(args.dev_file),
        "dev_sha256": file_sha256(args.dev_file),
        "reference_sha256": file_sha256(reference_file),
        "base_url": args.base_url,
        "served_model": args.model,
        "base_prompt_variant": PROMPT_VARIANT,
        "temperature": 0,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "workers": args.workers,
        "timeout_seconds": args.timeout_seconds,
        "arms": files,
        "arm_seconds": arm_seconds,
        "single_variable": "response_format_json_schema",
    }
    manifest_file = args.output_dir / "run_manifest.json"
    write_json_atomic(manifest_file, manifest)
    emit(
        "p1_structured_decoding_inference_complete",
        status="PASS",
        manifest_file=str(manifest_file),
        manifest_sha256=file_sha256(manifest_file),
        arm_seconds=arm_seconds,
        prediction_files=files,
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
        "--output-dir", type=Path,
        default=Path("reports/baselines/qwen2_5_7b_dev/v5_structured_decoding"),
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        emit(
            "p1_structured_decoding_inference_complete",
            status="FAIL",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
