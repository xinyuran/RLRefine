"""Audit SFT prompt/target alignment without training or generation."""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from prompts.prompt_template_3 import get_keyword_extraction_prompt_3
from scripts.canonicalize_sft_data import extract_role_content, extract_source_text


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True), flush=True)


def percentile(values: Sequence[int], probability: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def summarize(values: Sequence[int]) -> Dict[str, Any]:
    if not values:
        raise ValueError("Cannot summarize an empty sequence")
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def audit_rows(
    rows: Sequence[Dict[str, Any]],
    token_count: Callable[[str], int],
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("SFT rows must not be empty")
    assistant_chars: List[int] = []
    assistant_tokens: List[int] = []
    json_start_tokens: List[int] = []
    json_only_tokens: List[int] = []
    system_hashes = set()
    counts = {
        "think_tag_targets": 0,
        "assistant_over_max_new_tokens": 0,
        "json_starts_at_or_after_max_new_tokens": 0,
        "json_only_over_max_new_tokens": 0,
        "exact_b1_system_prompt": 0,
        "exact_b1_user_prompt": 0,
        "exact_b1_full_prompt": 0,
    }
    for row_number, row in enumerate(rows, 1):
        messages = row.get("messages")
        system = extract_role_content(messages, "system")
        user = extract_role_content(messages, "user")
        assistant = extract_role_content(messages, "assistant")
        source = extract_source_text(user)
        if not system or not user or not assistant or not source:
            raise ValueError(f"Incomplete SFT messages at row {row_number}")
        json_start = assistant.find("{")
        json_end = assistant.rfind("}")
        if json_start < 0 or json_end < json_start:
            raise ValueError(f"Missing target JSON at row {row_number}")
        expected_system, expected_user = get_keyword_extraction_prompt_3(source)
        system_match = system == expected_system
        user_match = user == expected_user
        counts["exact_b1_system_prompt"] += system_match
        counts["exact_b1_user_prompt"] += user_match
        counts["exact_b1_full_prompt"] += system_match and user_match
        counts["think_tag_targets"] += "<think>" in assistant and "</think>" in assistant
        system_hashes.add(hashlib.sha256(system.encode("utf-8")).hexdigest())

        total_tokens = token_count(assistant)
        start_tokens = token_count(assistant[:json_start])
        only_tokens = token_count(assistant[json_start:json_end + 1])
        assistant_chars.append(len(assistant))
        assistant_tokens.append(total_tokens)
        json_start_tokens.append(start_tokens)
        json_only_tokens.append(only_tokens)
        counts["assistant_over_max_new_tokens"] += total_tokens > max_new_tokens
        counts["json_starts_at_or_after_max_new_tokens"] += start_tokens >= max_new_tokens
        counts["json_only_over_max_new_tokens"] += only_tokens > max_new_tokens

    total = len(rows)
    return {
        "rows": total,
        "max_new_tokens": max_new_tokens,
        "system_prompt_variant_count": len(system_hashes),
        "counts": counts,
        "rates": {name: count / total for name, count in counts.items()},
        "assistant_chars": summarize(assistant_chars),
        "assistant_tokens": summarize(assistant_tokens),
        "json_start_tokens": summarize(json_start_tokens),
        "json_only_tokens": summarize(json_only_tokens),
    }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/canonical/keyword_v1/splits/sft_train.jsonl"))
    parser.add_argument("--dev-file", type=Path, default=Path("data/canonical/keyword_v1/splits/sft_dev.jsonl"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, default=Path("reports/diagnostics/p1_sft_v1_target_contract.json"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        token_count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
        report = {
            "report_version": "sft-target-contract-audit-v1",
            "label": "training_data_diagnostic_not_model_evaluation",
            "model_path": str(args.model_path),
            "train": audit_rows(read_jsonl(args.train_file), token_count, args.max_new_tokens),
            "dev": audit_rows(read_jsonl(args.dev_file), token_count, args.max_new_tokens),
        }
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        emit("p1_sft_target_contract_audit_complete", status="PASS", report_file=str(args.output_file), report=report)
        return 0
    except Exception as exc:
        emit("p1_sft_target_contract_audit_complete", status="FAIL", error_type=type(exc).__name__, error=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
