"""Deterministic offline vLLM replay for the 512→768 structured-output budget."""
import argparse
import hashlib
import json
import os
import time
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
from evaluation.structured_decoding_inference import (
    MAX_INPUT_TOKENS,
    PROMPT_VARIANT,
    REFERENCE_SHA256,
    keyword_response_format,
)
from evaluation.structured_decoding_token_budget_inference import (
    ORIGINAL_CONSTRAINED_SHA256,
    ORIGINAL_CONTROL_SHA256,
)


REPLAY_VARIANT = "b1_vllm_offline_json_schema_512_replay_v1"
CANDIDATE_VARIANT = "b1_vllm_offline_json_schema_768_v1"
EXPERIMENT_VERSION = "p1-structured-decoding-offline-token-budget-v1"


def build_chat_prompts(tokenizer: Any, rows: Sequence[Dict[str, Any]]) -> List[str]:
    prompts = []
    for row in rows:
        system_prompt, user_prompt = build_prompt(PROMPT_VARIANT, row["normalized_text"])
        prompts.append(tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        ))
    return prompts


def run_arm(
    llm: Any,
    prompts: Sequence[str],
    rows: Sequence[Dict[str, Any]],
    variant: str,
    max_tokens: int,
) -> tuple[List[Dict[str, Any]], float]:
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    schema = keyword_response_format()["json_schema"]["schema"]
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        seed=42,
        structured_outputs=StructuredOutputsParams(json=schema),
    )
    started = time.perf_counter()
    outputs = llm.generate(
        list(prompts),
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    arm_seconds = time.perf_counter() - started
    if len(outputs) != len(rows):
        raise ValueError(f"Unexpected output count for {variant}: {len(outputs)}")
    average_seconds = arm_seconds / len(rows)
    predictions = []
    for row, output in zip(rows, outputs):
        if len(output.outputs) != 1:
            raise ValueError(
                f"Unexpected completion count for {row['sample_id']}: {len(output.outputs)}"
            )
        completion = output.outputs[0]
        input_tokens = len(output.prompt_token_ids)
        output_tokens = len(completion.token_ids)
        if input_tokens > MAX_INPUT_TOKENS:
            raise ValueError(
                f"Prompt exceeds frozen {MAX_INPUT_TOKENS}-token budget for "
                f"{row['sample_id']}: {input_tokens}"
            )
        prediction = parse_completion(row["sample_id"], variant, completion.text)
        prediction.update({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "finish_reason": completion.finish_reason,
            "request_seconds": average_seconds,
            "experiment_version": EXPERIMENT_VERSION,
            "base_prompt_variant": PROMPT_VARIANT,
            "structured_output": True,
            "inference_mode": "vllm_offline_deterministic_scheduling",
        })
        predictions.append(prediction)
    return sorted(predictions, key=lambda item: item["sample_id"]), arm_seconds


def run(args: argparse.Namespace) -> int:
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise ValueError("VLLM_ENABLE_V1_MULTIPROCESSING must be 0")
    from vllm import LLM

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

    emit(
        "p1_structured_decoding_offline_model_load_start",
        model_path=str(args.model_path),
        v1_multiprocessing=os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"],
    )
    llm = LLM(
        model=str(args.model_path),
        dtype="bfloat16",
        seed=42,
        max_model_len=4864,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    prompts = build_chat_prompts(tokenizer, reference)
    emit("p1_structured_decoding_offline_model_load_complete", prompts=len(prompts))

    arm_specs = (
        (REPLAY_VARIANT, 512),
        (CANDIDATE_VARIANT, 768),
    )
    files: Dict[str, Dict[str, str]] = {}
    arm_seconds: Dict[str, float] = {}
    for variant, max_tokens in arm_specs:
        predictions, elapsed = run_arm(
            llm, prompts, reference, variant, max_tokens
        )
        arm_seconds[variant] = elapsed
        prediction_file = args.output_dir / f"{variant}_predictions.jsonl"
        write_jsonl_atomic(prediction_file, predictions)
        files[variant] = {
            "path": str(prediction_file),
            "sha256": file_sha256(prediction_file),
        }

    manifest: Dict[str, Any] = {
        "experiment_version": EXPERIMENT_VERSION,
        "label": "teacher_dev_offline_deterministic_token_budget_not_human_gold_test",
        "model_path": str(args.model_path),
        "dev_file": str(args.dev_file),
        "dev_sha256": file_sha256(args.dev_file),
        "reference_sha256": file_sha256(reference_file),
        "prior_dir": str(args.prior_dir),
        "prior_hashes": expected_hashes,
        "base_prompt_variant": PROMPT_VARIANT,
        "response_format": "same_json_schema_both_arms",
        "temperature": 0,
        "seed": 42,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_model_len": 4864,
        "dtype": "bfloat16",
        "vllm_enable_v1_multiprocessing": "0",
        "inference_mode": "offline",
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
        "p1_structured_decoding_offline_token_budget_inference_complete",
        status="PASS",
        manifest_file=str(manifest_file),
        manifest_sha256=file_sha256(manifest_file),
        prediction_files=files,
        arm_seconds=arm_seconds,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Local Base model directory; no private server path is assumed.",
    )
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
        default=Path("reports/baselines/qwen2_5_7b_dev/v7_offline_deterministic_768"),
    )
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        emit(
            "p1_structured_decoding_offline_token_budget_inference_complete",
            status="FAIL",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
