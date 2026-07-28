"""One-shot frozen-human-gold report for the preselected DPO and E4 models."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

from evaluation.baseline_inference import HUMAN_GOLD_SHA256, build_prompt, parse_completion, prepare_human_gold, stable_hash
from evaluation.keyword_evaluator import evaluate_records, file_sha256, read_jsonl, render_markdown, write_json_atomic, write_jsonl_atomic, write_text_atomic


GOLD_ROWS = 496
PROMPT_VARIANT = "b1_structured_v3"


def same_adapter_path(expected: Path, received: Path) -> bool:
    return expected.resolve() == received.resolve()


def load_freeze(path: Path) -> dict[str, Any]:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("decision") != "FREEZE_E4_AS_OPTIMAL_RESEARCH_CANDIDATE" or freeze.get("status") != "PASS":
        raise ValueError("E4 research-candidate freeze is not valid")
    if freeze.get("human_gold_used_for_selection") is not False:
        raise ValueError("Candidate freeze must not use human gold")
    return freeze


def _run_model(model: Any, tokenizer: Any, rows: list[dict[str, Any]], output: Path, model_id: str, batch_size: int, max_input_tokens: int, max_new_tokens: int) -> list[dict[str, Any]]:
    import torch
    completed = {row["sample_id"]: row for row in read_jsonl(output)} if output.is_file() else {}
    allowed = {row["sample_id"] for row in rows}
    if set(completed) - allowed:
        raise ValueError(f"Unexpected resumable predictions in {output}")
    for start in range(0, len(rows), batch_size):
        batch = [row for row in rows[start:start + batch_size] if row["sample_id"] not in completed]
        if not batch:
            continue
        conversations = [[{"role": "system", "content": build_prompt(PROMPT_VARIANT, row["normalized_text"])[0]}, {"role": "user", "content": build_prompt(PROMPT_VARIANT, row["normalized_text"])[1]}] for row in batch]
        prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in conversations]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_tokens).to(model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        elapsed = time.perf_counter() - started
        generated_only = generated[:, inputs["input_ids"].shape[1]:]
        for row, response, attention in zip(batch, tokenizer.batch_decode(generated_only, skip_special_tokens=True), inputs["attention_mask"]):
            prediction = parse_completion(row["sample_id"], PROMPT_VARIANT, response)
            prediction.update({"model_id": model_id, "input_tokens": int(attention.sum().item()), "output_tokens": len(tokenizer.encode(response, add_special_tokens=False)), "request_seconds": elapsed / len(batch)})
            completed[row["sample_id"]] = prediction
        write_jsonl_atomic(output, [completed[row["sample_id"]] for row in rows if row["sample_id"] in completed])
    if set(completed) != allowed:
        raise ValueError(f"Incomplete one-shot predictions for {model_id}")
    return [completed[row["sample_id"]] for row in rows]


def _write_report(output_dir: Path, model_id: str, gold: list[dict[str, Any]], predictions: list[dict[str, Any]], max_new_tokens: int) -> dict[str, Any]:
    report, errors = evaluate_records(gold, predictions, model_id)
    report.update({"label": "frozen_human_gold_one_shot_not_for_model_selection", "evaluation_policy": "one_shot_after_e4_candidate_freeze_no_gold_tuning", "resource_metrics": {"mean_input_tokens": sum(row["input_tokens"] for row in predictions) / len(predictions), "mean_output_tokens": sum(row["output_tokens"] for row in predictions) / len(predictions), "mean_request_seconds": sum(row["request_seconds"] for row in predictions) / len(predictions), "max_token_output_count": sum(row["output_tokens"] >= max_new_tokens for row in predictions)}})
    write_json_atomic(output_dir / f"{model_id}_evaluation.json", report)
    write_jsonl_atomic(output_dir / f"{model_id}_errors.jsonl", errors)
    write_text_atomic(output_dir / f"{model_id}_evaluation.md", render_markdown(report))
    return report


def run(args: argparse.Namespace) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (args.batch_size, args.max_input_tokens, args.max_new_tokens, args.seed) != (4, 4096, 768, 42):
        raise ValueError("One-shot protocol is frozen to batch=4, input=4096, output=768, seed=42")
    freeze = load_freeze(args.e4_freeze)
    if file_sha256(args.gold_file) != HUMAN_GOLD_SHA256:
        raise ValueError("Frozen human gold hash mismatch")
    gold = prepare_human_gold(read_jsonl(args.gold_file))
    if len(gold) != GOLD_ROWS:
        raise ValueError("Frozen human gold row count mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for human-gold one-shot inference")
    e4_checkpoint = Path(freeze["candidate_checkpoint"])
    if not same_adapter_path(e4_checkpoint, args.e4_adapter):
        raise ValueError(
            "E4 adapter must equal the frozen research candidate checkpoint: "
            f"expected={e4_checkpoint.resolve()}, received={args.e4_adapter.resolve()}"
        )
    for adapter in (args.dpo_adapter, args.e4_adapter):
        if not (adapter / "adapter_config.json").is_file() or not (adapter / "adapter_model.safetensors").is_file():
            raise ValueError(f"Incomplete adapter: {adapter}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"report_version": "bp5-dpo-e4-human-gold-one-shot-v1", "gold_sha256": file_sha256(args.gold_file), "gold_rows": len(gold), "gold_sample_ids_sha256": stable_hash("\n".join(row["sample_id"] for row in gold)), "base_model": str(args.base_model), "dpo_adapter": str(args.dpo_adapter), "e4_adapter": str(args.e4_adapter), "e4_freeze_sha256": file_sha256(args.e4_freeze), "prompt_variant": PROMPT_VARIANT, "decoding": {"do_sample": False, "seed": 42, "max_input_tokens": 4096, "max_new_tokens": 768}, "evaluation_policy": "one_shot_after_e4_candidate_freeze_no_gold_tuning", "human_gold_used_for_selection": False}
    manifest_path = args.output_dir / "run_manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Existing one-shot manifest differs from preregistered protocol")
    write_json_atomic(manifest_path, manifest)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    reports = {}
    for model_id, adapter in (("E2_DPO_RELIABILITY", args.dpo_adapter), ("E4_GRPO_FROM_DPO", args.e4_adapter)):
        base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, adapter).cuda().eval()
        predictions = _run_model(model, tokenizer, gold, args.output_dir / f"{model_id}_predictions.jsonl", model_id, 4, 4096, 768)
        reports[model_id] = _write_report(args.output_dir, model_id, gold, predictions, 768)
        del model, base
        torch.cuda.empty_cache()
    summary = {"status": "PASS", "event": "bp5_dpo_e4_human_gold_one_shot_complete", "manifest_sha256": file_sha256(manifest_path), "human_gold_used_for_selection": False, "reports": {key: {"macro_f1": value["macro"]["f1"], "micro_f1": value["micro"]["f1"], "schema_valid_rate": value["schema_valid_rate"]} for key, value in reports.items()}, "next_execution": "BP5_SCHEMA_ERROR_TAXONOMY_AND_SERVING_BENCHMARK"}
    write_json_atomic(args.output_dir / "one_shot_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True); parser.add_argument("--dpo-adapter", type=Path, required=True); parser.add_argument("--e4-adapter", type=Path, required=True); parser.add_argument("--e4-freeze", type=Path, required=True); parser.add_argument("--gold-file", type=Path, default=Path("data/canonical/keyword_v1/gold/v1/gold_test.jsonl")); parser.add_argument("--output-dir", type=Path, default=Path("reports/bp5/human_gold_one_shot")); parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--max-input-tokens", type=int, default=4096); parser.add_argument("--max-new-tokens", type=int, default=768); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try: return run(args)
    except Exception as exc:
        print(json.dumps({"event": "bp5_dpo_e4_human_gold_one_shot_complete", "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, sort_keys=True)); return 1


if __name__ == "__main__":
    raise SystemExit(main())
