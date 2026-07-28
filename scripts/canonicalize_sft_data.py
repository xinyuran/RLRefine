"""Canonicalize SFT references and quarantine unsafe records.

Safe automatic repair is intentionally limited to numeric strings in the
confidence field. Records requiring semantic judgment are quarantined with
explicit reasons instead of being silently changed or discarded.
"""

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


CONTRACT_VERSION = "keyword-v1"
COMMENT_START = "【待处理评论】"
COMMENT_END = "请严格"


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def extract_role_content(messages: Any, role: str) -> str:
    if not isinstance(messages, list):
        return ""
    return next(
        (
            item.get("content", "")
            for item in messages
            if isinstance(item, dict) and item.get("role") == role
        ),
        "",
    )


def extract_source_text(user_content: str) -> str:
    if not isinstance(user_content, str) or COMMENT_START not in user_content:
        return ""
    source = user_content.split(COMMENT_START, 1)[1]
    if COMMENT_END in source:
        source = source.split(COMMENT_END, 1)[0]
    return source.strip()


def parse_response_json(response: str) -> Tuple[Optional[Dict[str, Any]], int, int, Optional[str]]:
    if not isinstance(response, str) or not response.strip():
        return None, -1, -1, "assistant_response_missing"
    start = response.find("{")
    end = response.rfind("}")
    if start < 0 or end < start:
        return None, start, end, "assistant_json_missing"
    try:
        parsed = json.loads(response[start:end + 1])
    except json.JSONDecodeError:
        return None, start, end, "assistant_json_invalid"
    if not isinstance(parsed, dict):
        return None, start, end, "assistant_json_not_object"
    return parsed, start, end, None


def stable_sample_id(source_text: str) -> str:
    normalized = re.sub(r"\s+", "", source_text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def canonicalize_keyword_items(
    parsed: Dict[str, Any],
    source_text: str,
) -> Tuple[List[str], int]:
    reasons: Set[str] = set()
    converted_string_scores = 0

    if set(parsed) != {"keywords"}:
        reasons.add("unexpected_top_level_fields")

    keywords = parsed.get("keywords")
    if not isinstance(keywords, list):
        return ["keywords_not_array", *sorted(reasons)], converted_string_scores
    if not keywords:
        reasons.add("keywords_empty")
    if len(keywords) > 15:
        reasons.add("keywords_over_limit")

    for item in keywords:
        if not isinstance(item, list) or len(item) != 3:
            reasons.add("keyword_item_invalid_shape")
            continue

        explanation, keyword, score = item
        if not isinstance(explanation, str) or not explanation.strip():
            reasons.add("keyword_explanation_invalid")

        if not isinstance(keyword, str) or not keyword.strip():
            reasons.add("keyword_text_invalid")
        else:
            keyword = keyword.strip()
            item[1] = keyword
            if len(keyword) > 4:
                reasons.add("keyword_too_long")
            if keyword not in source_text:
                reasons.add("keyword_not_in_source")

        if isinstance(score, bool):
            reasons.add("confidence_invalid_type")
        elif isinstance(score, (int, float)):
            if not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
                reasons.add("confidence_out_of_range")
        elif isinstance(score, str):
            try:
                numeric_score = float(score.strip())
            except ValueError:
                reasons.add("confidence_invalid_string")
            else:
                if math.isfinite(numeric_score) and 0 <= numeric_score <= 1:
                    item[2] = numeric_score
                    converted_string_scores += 1
                else:
                    reasons.add("confidence_out_of_range")
        else:
            reasons.add("confidence_invalid_type")

    return sorted(reasons), converted_string_scores


def canonicalize_record(
    record: Dict[str, Any],
    line_number: int,
    seen_sample_ids: Set[str],
) -> Tuple[Optional[Dict[str, Any]], List[str], int, Optional[str]]:
    reasons: Set[str] = set()
    messages = record.get("messages") if isinstance(record, dict) else None
    if not isinstance(messages, list):
        return None, ["messages_invalid"], 0, None

    user_content = extract_role_content(messages, "user")
    assistant_content = extract_role_content(messages, "assistant")
    source_text = extract_source_text(user_content)
    if not source_text:
        reasons.add("source_text_missing")

    sample_id = stable_sample_id(source_text) if source_text else None
    parsed, start, end, parse_error = parse_response_json(assistant_content)
    if parse_error:
        reasons.add(parse_error)

    converted_string_scores = 0
    if parsed is not None and source_text:
        contract_reasons, converted_string_scores = canonicalize_keyword_items(
            parsed,
            source_text,
        )
        reasons.update(contract_reasons)

    if sample_id and sample_id in seen_sample_ids:
        reasons.add("duplicate_input")

    if reasons:
        return None, sorted(reasons), converted_string_scores, sample_id

    canonical_messages = [dict(item) for item in messages]
    canonical_response = (
        assistant_content[:start]
        + json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        + assistant_content[end + 1:]
    )
    for message in canonical_messages:
        if message.get("role") == "assistant":
            message["content"] = canonical_response
            break

    seen_sample_ids.add(sample_id)
    return {
        "sample_id": sample_id,
        "group_id": sample_id,
        "source_line": line_number,
        "contract_version": CONTRACT_VERSION,
        "messages": canonical_messages,
    }, [], converted_string_scores, sample_id


def read_jsonl(path: Path) -> Iterable[Tuple[int, Optional[Dict[str, Any]], Optional[str]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, None, str(exc)
            else:
                yield line_number, record, None


def write_jsonl_atomic(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def canonicalize_file(input_file: Path, output_dir: Path) -> Dict[str, Any]:
    clean_records: List[Dict[str, Any]] = []
    quarantine_records: List[Dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    seen_sample_ids: Set[str] = set()
    input_rows = 0
    converted_string_scores = 0

    emit("canonicalization_start", input_file=str(input_file), output_dir=str(output_dir))
    for line_number, record, jsonl_error in read_jsonl(input_file):
        input_rows += 1
        if jsonl_error:
            reasons = ["invalid_jsonl"]
            canonical = None
            sample_id = None
            converted = 0
        else:
            canonical, reasons, converted, sample_id = canonicalize_record(
                record,
                line_number,
                seen_sample_ids,
            )
        converted_string_scores += converted

        if canonical is not None:
            clean_records.append(canonical)
            continue

        reason_counts.update(reasons)
        quarantine_records.append({
            "source_line": line_number,
            "sample_id": sample_id,
            "reasons": reasons,
            "record": record,
        })

    clean_file = output_dir / "sft_clean.jsonl"
    quarantine_file = output_dir / "sft_quarantine.jsonl"
    report_file = output_dir / "canonicalization_report.json"
    write_jsonl_atomic(clean_file, clean_records)
    write_jsonl_atomic(quarantine_file, quarantine_records)

    report = {
        "event": "canonicalization_complete",
        "contract_version": CONTRACT_VERSION,
        "input_file": str(input_file),
        "input_rows": input_rows,
        "clean_rows": len(clean_records),
        "quarantine_rows": len(quarantine_records),
        "clean_rate": round(len(clean_records) / input_rows, 6) if input_rows else 0.0,
        "converted_string_scores": converted_string_scores,
        "reason_counts": dict(sorted(reason_counts.items())),
        "clean_file": str(clean_file),
        "quarantine_file": str(quarantine_file),
    }
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    emit(**report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.input_file.is_file():
        emit("canonicalization_complete", status="FAIL", error=f"File not found: {args.input_file}")
        return 1
    report = canonicalize_file(args.input_file, args.output_dir)
    return 0 if report["clean_rows"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
