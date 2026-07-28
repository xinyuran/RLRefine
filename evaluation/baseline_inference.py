"""B0/B1 base-model inference on the development split."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from evaluation.keyword_evaluator import (
    evaluate_records,
    file_sha256,
    read_jsonl,
    render_markdown,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from examples.keyword_extraction.schema import create_keyword_schema
from prompts.prompt_template_3 import get_keyword_extraction_prompt_3
from prompts.prompt_template_4 import get_keyword_extraction_prompt_4
from prompts.prompt_template_5 import get_keyword_extraction_prompt_5
from prompts.prompt_template_6 import get_keyword_extraction_prompt_6


DEV_SAMPLE_IDS_SHA256 = "6889e8e8b54dc7d1441776d18fadbcff9e0ece862fe68ee310a1b73a7d198d42"
HUMAN_GOLD_SHA256 = "0df4d88e8bc09d62f867ef8a4b43fe631ed9e357b79b55e022948f9b75e06bf6"
PROMPT_VERSION = "base-baseline-b0-b1-v1"
SUPPORTED_VARIANTS = (
    "b0_simple", "b1_structured_v3", "b2_compact_v1", "b3_balanced_v1",
    "sft_v2_json_only_protocol_v1",
)


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True), flush=True)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _extract_json_object(text: str) -> Dict[str, Any] | None:
    decoder = json.JSONDecoder()
    candidates = []
    for match in re.finditer(r"\{", text or ""):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    keyword_candidates = [value for value in candidates if "keywords" in value]
    if keyword_candidates:
        return keyword_candidates[-1]
    return candidates[-1] if candidates else None


def _source_from_messages(messages: Sequence[Dict[str, Any]]) -> str:
    user_text = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    )
    patterns = [
        r"【待处理评论】\s*\n(.+?)(?:\n\n请严格|$)",
        r"【待处理文本】\s*\n(.+?)(?:\n\n请严格|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_text, re.DOTALL)
        if match:
            return match.group(1).strip()
    raise ValueError("Unable to extract source text from dev messages")


def prepare_dev_gold(grpo_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    gold_rows = []
    seen = set()
    for row_number, row in enumerate(grpo_rows, 1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"Invalid or duplicate sample_id at dev row {row_number}")
        seen.add(sample_id)
        source_text = _source_from_messages(row.get("messages") or [])
        solution = _extract_json_object(row.get("solution") or "")
        valid, errors = create_keyword_schema().validate(solution or {})
        if not valid:
            raise ValueError(f"Invalid dev solution for {sample_id}: {errors}")
        keywords = [item[1] for item in solution["keywords"]]
        if len(keywords) != len(set(keywords)):
            raise ValueError(f"Duplicate dev solution keywords for {sample_id}")
        if any(keyword not in source_text for keyword in keywords):
            raise ValueError(f"Dev solution keyword is not in source for {sample_id}")
        gold_rows.append({
            "sample_id": sample_id,
            "normalized_text": source_text,
            "gold": {"keywords": keywords},
            "label_origin": "teacher_dev_reference_not_human_gold",
            "split": "dev",
        })
    return sorted(gold_rows, key=lambda item: item["sample_id"])


def prepare_human_gold(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared = []
    seen = set()
    for row_number, row in enumerate(rows, 1):
        sample_id = row.get("sample_id")
        source_text = row.get("normalized_text")
        gold = row.get("gold")
        keywords = gold.get("keywords") if isinstance(gold, dict) else None
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"Invalid or duplicate human gold sample_id at row {row_number}")
        seen.add(sample_id)
        if row.get("dataset_version") != "keyword-gold-v1" or row.get("split") != "test_gold":
            raise ValueError(f"Invalid human gold provenance for {sample_id}")
        if not isinstance(source_text, str) or not source_text:
            raise ValueError(f"Missing human gold source text for {sample_id}")
        if not isinstance(keywords, list) or not keywords or any(
            not isinstance(keyword, str) or not 1 <= len(keyword) <= 4
            for keyword in keywords
        ):
            raise ValueError(f"Invalid human gold keywords for {sample_id}")
        if len(keywords) != len(set(keywords)) or any(keyword not in source_text for keyword in keywords):
            raise ValueError(f"Human gold keywords violate source contract for {sample_id}")
        prepared.append(row)
    return sorted(prepared, key=lambda item: item["sample_id"])


def build_prompt(variant: str, source_text: str) -> Tuple[str, str]:
    if variant == "b0_simple":
        return (
            "你是中文电商评论关键词抽取助手。只输出合法JSON，不要输出解释。",
            "从下面评论提取1至15个原文关键词，每个关键词1至4个汉字。"
            "输出格式必须是{\"keywords\":[[\"类别\",\"关键词\",0.9]]}，"
            "置信度必须是0到1之间的数字。\n\n评论：\n" + source_text,
        )
    if variant == "b1_structured_v3":
        return get_keyword_extraction_prompt_3(source_text)
    if variant == "b2_compact_v1":
        return get_keyword_extraction_prompt_4(source_text)
    if variant == "b3_balanced_v1":
        return get_keyword_extraction_prompt_5(source_text)
    if variant == "sft_v2_json_only_protocol_v1":
        return get_keyword_extraction_prompt_6(source_text)
    raise ValueError(f"Unknown prompt variant: {variant}")


def parse_completion(sample_id: str, variant: str, raw_response: str) -> Dict[str, Any]:
    parsed = _extract_json_object(raw_response)
    if parsed is None:
        return {
            "sample_id": sample_id,
            "status": "error",
            "data": None,
            "error_code": "json_parse_failed",
            "prompt_variant": variant,
            "raw_response": raw_response,
        }
    valid, errors = create_keyword_schema().validate(parsed)
    if not valid:
        return {
            "sample_id": sample_id,
            "status": "error",
            "data": None,
            "error_code": "schema_validation_failed",
            "validation_errors": errors,
            "prompt_variant": variant,
            "raw_response": raw_response,
        }
    return {
        "sample_id": sample_id,
        "status": "success",
        "data": parsed,
        "prompt_variant": variant,
        "raw_response": raw_response,
    }


def _load_existing(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        return {}
    rows = read_jsonl(path)
    indexed = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in indexed:
            raise ValueError(f"Invalid resumable prediction file: {path}")
        indexed[sample_id] = row
    return indexed


def _write_variant_report(
    output_dir: Path,
    variant: str,
    dev_gold: Sequence[Dict[str, Any]],
    predictions: Sequence[Dict[str, Any]],
    model_id: str,
    prompt_version: str = PROMPT_VERSION,
    label: str = "teacher_dev_baseline_not_human_gold_test",
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    report, errors = evaluate_records(dev_gold, predictions, f"{model_id}:{variant}")
    report["label"] = label
    report["prompt_variant"] = variant
    report["prompt_version"] = prompt_version
    report_file = output_dir / f"{variant}_evaluation.json"
    markdown_file = output_dir / f"{variant}_evaluation.md"
    error_file = output_dir / f"{variant}_errors.jsonl"
    write_json_atomic(report_file, report)
    write_text_atomic(markdown_file, render_markdown(report))
    write_jsonl_atomic(error_file, errors)
    return {
        "variant": variant,
        "prediction_file": str(output_dir / f"{variant}_predictions.jsonl"),
        "prediction_sha256": file_sha256(output_dir / f"{variant}_predictions.jsonl"),
        "report_file": str(report_file),
        "report_sha256": file_sha256(report_file),
        "markdown_file": str(markdown_file),
        "markdown_sha256": file_sha256(markdown_file),
        "errors_file": str(error_file),
        "errors_sha256": file_sha256(error_file),
        "metrics": {
            "schema_valid_rate": report["schema_valid_rate"],
            "exact_set_match_rate": report["exact_set_match_rate"],
            "micro_f1": report["micro"]["f1"],
            "macro_f1": report["macro"]["f1"],
            "hallucination_keyword_rate": report["hallucination"]["keyword_rate"],
            "mean_input_tokens": sum(row["input_tokens"] for row in predictions) / len(predictions),
            "mean_output_tokens": sum(row["output_tokens"] for row in predictions) / len(predictions),
            "max_token_output_count": sum(
                row["output_tokens"] >= max_new_tokens for row in predictions
            ),
            "status_counts": report["status_counts"],
            "error_sample_count": report["error_sample_count"],
        },
    }


def run(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = getattr(args, "adapter_path", None)
    is_sft_candidate = adapter_path is not None
    if args.batch_size < 1 or args.max_input_tokens < 1 or args.max_new_tokens < 1:
        raise ValueError("batch-size and token limits must all be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen2.5-7B baseline run")

    gold_file = getattr(args, "gold_file", None)
    if gold_file is not None:
        if file_sha256(gold_file) != HUMAN_GOLD_SHA256:
            raise ValueError("Human gold file hash mismatch")
        dev_gold = prepare_human_gold(read_jsonl(gold_file))
        if len(dev_gold) != 496:
            raise ValueError(f"Unexpected human gold row count: {len(dev_gold)}")
        source_file = gold_file
        reference_label = "frozen_human_gold_test"
        evaluation_label = "final_base_evaluation_on_frozen_human_gold"
    else:
        grpo_rows = read_jsonl(args.dev_file)
        dev_gold = prepare_dev_gold(grpo_rows)
        source_file = args.dev_file
        reference_label = "teacher_dev_reference_not_human_gold"
        evaluation_label = (
            "sft_candidate_teacher_dev_selection_not_human_gold_test"
            if is_sft_candidate else "teacher_dev_baseline_not_human_gold_test"
        )
    sample_ids_hash = stable_hash("\n".join(row["sample_id"] for row in dev_gold))
    if gold_file is None and (len(dev_gold) != 331 or sample_ids_hash != DEV_SAMPLE_IDS_SHA256):
        raise ValueError(
            f"Unexpected dev split: rows={len(dev_gold)}, sample_ids_sha256={sample_ids_hash}"
        )

    variants = list(getattr(args, "variants", None) or ["b0_simple", "b1_structured_v3"])
    if len(variants) != len(set(variants)) or any(item not in SUPPORTED_VARIANTS for item in variants):
        raise ValueError(f"Invalid prompt variants: {variants}")
    if gold_file is not None and (
        variants != ["b1_structured_v3"]
        or args.max_input_tokens != 4096
        or args.max_new_tokens != 512
        or args.seed != 42
    ):
        raise ValueError("Human gold evaluation requires frozen B1 and decoding configuration")
    if is_sft_candidate and (
        gold_file is not None
        or variants != ["b1_structured_v3"]
        or args.max_input_tokens != 4096
        or args.max_new_tokens != 512
        or args.seed != 42
    ):
        raise ValueError("SFT candidate evaluation requires frozen teacher-dev B1 decoding")
    prompt_version = (
        PROMPT_VERSION
        if variants == ["b0_simple", "b1_structured_v3"]
        else f"prompt-optimization-{variants[0].split('_', 1)[0]}-v1"
        if len(variants) == 1
        else "prompt-optimization-custom-v1"
    )
    if gold_file is not None:
        prompt_version = "frozen-b1-structured-v3-v1"
    config = {
        "run_version": (
            "sft-teacher-dev-v1" if is_sft_candidate
            else "final-base-human-gold-v1" if gold_file is not None
            else "base-baseline-v1"
        ),
        "model_path": str(args.model_path),
        "prompt_version": prompt_version,
        "variants": variants,
        "batch_size": args.batch_size,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "seed": args.seed,
    }
    if is_sft_candidate:
        if adapter_path.name != "checkpoint-390":
            raise ValueError("SFT candidate must be the selected checkpoint-390")
        required_adapter_files = (
            "adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
        )
        missing = [name for name in required_adapter_files if not (adapter_path / name).is_file()]
        if missing:
            raise ValueError(f"SFT adapter checkpoint is incomplete: missing={missing}")
        trainer_state = json.loads((adapter_path / "trainer_state.json").read_text(encoding="utf-8"))
        if (
            Path(trainer_state.get("best_model_checkpoint") or "").name != "checkpoint-390"
            or trainer_state.get("global_step") != 390
        ):
            raise ValueError("SFT trainer state does not select completed checkpoint-390")
        config.update({
            "adapter_path": str(adapter_path),
            "adapter_artifact_sha256": {
                name: file_sha256(adapter_path / name) for name in required_adapter_files
            },
            "selection_policy": "best_training_eval_loss_then_frozen_teacher_dev_acceptance",
        })
    if gold_file is None:
        config.update({
            "dev_file": str(source_file),
            "dev_sha256": file_sha256(source_file),
            "dev_sample_ids_sha256": sample_ids_hash,
        })
    else:
        config.update({
            "gold_file": str(source_file),
            "gold_sha256": file_sha256(source_file),
            "gold_sample_ids_sha256": sample_ids_hash,
            "evaluation_policy": "one_shot_after_prompt_freeze_no_gold_tuning",
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = args.output_dir / "run_manifest.json"
    if manifest_file.is_file():
        existing_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if existing_manifest != config:
            raise ValueError("Existing baseline manifest does not match requested configuration")
    else:
        write_json_atomic(manifest_file, config)
    dev_reference_file = args.output_dir / (
        "human_gold_reference.jsonl" if gold_file is not None else "dev_teacher_reference.jsonl"
    )
    write_jsonl_atomic(dev_reference_file, dev_gold)
    emit(
        "baseline_gold_contract" if gold_file is not None else "baseline_dev_contract",
        rows=len(dev_gold),
        source_file=str(source_file),
        source_sha256=file_sha256(source_file),
        sample_ids_sha256=sample_ids_hash,
        reference_file=str(dev_reference_file),
        reference_sha256=file_sha256(dev_reference_file),
        label=reference_label,
    )

    emit("baseline_model_load_start", model_path=str(args.model_path))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda().eval()
    if is_sft_candidate:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path).eval()
    emit("baseline_model_load_complete", device=str(model.device), dtype=str(model.dtype))

    summaries = []
    for variant in config["variants"]:
        prediction_file = args.output_dir / f"{variant}_predictions.jsonl"
        completed = _load_existing(prediction_file)
        unknown_ids = sorted(set(completed) - {row["sample_id"] for row in dev_gold})
        wrong_variants = sorted(
            sample_id
            for sample_id, row in completed.items()
            if row.get("prompt_variant") != variant
        )
        if unknown_ids or wrong_variants:
            raise ValueError(
                "Existing predictions do not match this run: "
                f"unknown_ids={len(unknown_ids)}, wrong_variants={len(wrong_variants)}"
            )
        pending = [row for row in dev_gold if row["sample_id"] not in completed]
        emit(
            "baseline_variant_start",
            variant=variant,
            total=len(dev_gold),
            resumed=len(completed),
            pending=len(pending),
        )
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            conversations = []
            for row in batch:
                system_prompt, user_prompt = build_prompt(variant, row["normalized_text"])
                conversations.append([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ])
            prompt_texts = [
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in conversations
            ]
            inputs = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_tokens,
            ).to(model.device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.perf_counter() - started
            generated_only = generated[:, inputs["input_ids"].shape[1]:]
            responses = tokenizer.batch_decode(generated_only, skip_special_tokens=True)
            for row, response, attention_mask in zip(
                batch, responses, inputs["attention_mask"]
            ):
                prediction = parse_completion(row["sample_id"], variant, response)
                prediction["input_tokens"] = int(attention_mask.sum().item())
                prediction["output_tokens"] = len(
                    tokenizer.encode(response, add_special_tokens=False)
                )
                completed[row["sample_id"]] = prediction
            ordered = [completed[row["sample_id"]] for row in dev_gold if row["sample_id"] in completed]
            write_jsonl_atomic(prediction_file, ordered)
            emit(
                "baseline_batch_complete",
                variant=variant,
                completed=len(completed),
                total=len(dev_gold),
                batch_rows=len(batch),
                batch_seconds=round(elapsed, 3),
            )

        predictions = [completed[row["sample_id"]] for row in dev_gold]
        summaries.append(
            _write_variant_report(
                args.output_dir,
                variant,
                dev_gold,
                predictions,
                args.model_id,
                prompt_version,
                evaluation_label,
                args.max_new_tokens,
            )
        )
        emit("baseline_variant_complete", **summaries[-1])

    summary = {
        "event": (
            "p1_sft_teacher_dev_complete" if is_sft_candidate
            else "p1_final_base_gold_complete" if gold_file is not None
            else "p1_base_baseline_complete"
        ),
        "status": "PASS",
        "label": evaluation_label,
        "model_id": args.model_id,
        "manifest_file": str(manifest_file),
        "manifest_sha256": file_sha256(manifest_file),
        "variants": summaries,
    }
    if gold_file is None:
        summary.update({
            "dev_rows": len(dev_gold),
            "dev_sample_ids_sha256": sample_ids_hash,
            "dev_reference_file": str(dev_reference_file),
            "dev_reference_sha256": file_sha256(dev_reference_file),
        })
    else:
        summary.update({
            "gold_rows": len(dev_gold),
            "gold_sample_ids_sha256": sample_ids_hash,
            "gold_reference_file": str(dev_reference_file),
            "gold_reference_sha256": file_sha256(dev_reference_file),
            "evaluation_policy": "one_shot_after_prompt_freeze_no_gold_tuning",
        })
    write_json_atomic(
        args.output_dir / ("final_gold_summary.json" if gold_file is not None else "baseline_summary.json"),
        summary,
    )
    emit(**summary)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen2.5-7B-Instruct-base")
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--dev-file", type=Path, default=Path("data/canonical/keyword_v1/splits/grpo_dev.jsonl"))
    parser.add_argument("--gold-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/baselines/qwen2_5_7b_dev/v1"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", nargs="+", choices=SUPPORTED_VARIANTS)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        emit(
            "p1_sft_teacher_dev_complete" if args.adapter_path is not None
            else "p1_final_base_gold_complete" if args.gold_file is not None
            else "p1_base_baseline_complete",
            status="FAIL", error_type=type(exc).__name__, error=str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
