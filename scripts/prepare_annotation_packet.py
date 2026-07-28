"""Prepare blinded primary/secondary annotation packets for test candidates."""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.canonicalize_sft_data import (  # noqa: E402
    extract_role_content,
    extract_source_text,
    parse_response_json,
)
from scripts.split_canonical_data import file_sha256, stable_hash  # noqa: E402


ANNOTATION_VERSION = "keyword-gold-v1-annotation-v1"
DEFAULT_SEED = "structalign-keyword-gold-v1"
ANNOTATION_COLUMNS = (
    "sample_id",
    "source_text",
    "annotator_id",
    "keywords_json",
    "annotation_status",
    "exclude_reason",
    "notes",
)


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            records.append(record)
    return records


def extract_candidate(record: Dict[str, Any]) -> Dict[str, Any]:
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("Every test candidate must have a non-empty sample_id")
    if record.get("split") != "test_candidate":
        raise ValueError(f"Sample {sample_id} is not marked test_candidate")

    messages = record.get("messages")
    user_content = extract_role_content(messages, "user")
    assistant_content = extract_role_content(messages, "assistant")
    source_text = extract_source_text(user_content)
    if not source_text:
        raise ValueError(f"Sample {sample_id} has no extractable source text")

    parsed, _, _, error = parse_response_json(assistant_content)
    if error or not isinstance(parsed.get("keywords"), list):
        raise ValueError(f"Sample {sample_id} has invalid teacher reference: {error}")
    teacher_keywords = [item[1] for item in parsed["keywords"]]

    return {
        "sample_id": sample_id,
        "group_id": record.get("group_id") or sample_id,
        "source_line": record.get("source_line"),
        "source_text": source_text,
        "teacher_keywords": teacher_keywords,
    }


def prepare_rows(
    records: Sequence[Dict[str, Any]],
    secondary_ratio: float = 0.20,
    seed: str = DEFAULT_SEED,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, Any]]]:
    if not records:
        raise ValueError("Cannot prepare an annotation packet from an empty dataset")
    if not 0 < secondary_ratio <= 1:
        raise ValueError("secondary_ratio must be in (0, 1]")

    candidates = [extract_candidate(record) for record in records]
    sample_ids = [item["sample_id"] for item in candidates]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample_id in test candidates")

    ordered = sorted(
        candidates,
        key=lambda item: stable_hash(f"{seed}:primary:{item['sample_id']}"),
    )
    secondary_count = math.ceil(len(ordered) * secondary_ratio)
    secondary_ids = {
        item["sample_id"]
        for item in sorted(
            candidates,
            key=lambda item: stable_hash(f"{seed}:secondary:{item['sample_id']}"),
        )[:secondary_count]
    }

    def blind_row(item: Dict[str, Any]) -> Dict[str, str]:
        return {
            "sample_id": item["sample_id"],
            "source_text": item["source_text"],
            "annotator_id": "",
            "keywords_json": "",
            "annotation_status": "",
            "exclude_reason": "",
            "notes": "",
        }

    primary_rows = [blind_row(item) for item in ordered]
    secondary_rows = [blind_row(item) for item in ordered if item["sample_id"] in secondary_ids]
    teacher_rows = [
        {
            "sample_id": item["sample_id"],
            "group_id": item["group_id"],
            "source_line": item["source_line"],
            "teacher_keywords": item["teacher_keywords"],
        }
        for item in ordered
    ]
    return primary_rows, secondary_rows, teacher_rows


def write_csv_atomic(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def prepare_packet(
    input_file: Path,
    output_dir: Path,
    secondary_ratio: float,
    seed: str,
) -> Dict[str, Any]:
    emit(
        "annotation_packet_start",
        input_file=str(input_file),
        output_dir=str(output_dir),
        secondary_ratio=secondary_ratio,
        seed=seed,
    )
    records = read_jsonl(input_file)
    primary_rows, secondary_rows, teacher_rows = prepare_rows(
        records,
        secondary_ratio,
        seed,
    )

    primary_file = output_dir / "annotation_primary.csv"
    secondary_file = output_dir / "annotation_secondary.csv"
    teacher_file = output_dir / "review_only_teacher_reference.jsonl"
    write_csv_atomic(primary_file, primary_rows)
    write_csv_atomic(secondary_file, secondary_rows)
    write_jsonl_atomic(teacher_file, teacher_rows)

    secondary_ids = sorted(row["sample_id"] for row in secondary_rows)
    manifest = {
        "event": "annotation_packet_complete",
        "annotation_version": ANNOTATION_VERSION,
        "seed": seed,
        "source_file": str(input_file),
        "source_sha256": file_sha256(input_file),
        "primary_rows": len(primary_rows),
        "secondary_rows": len(secondary_rows),
        "secondary_ratio_requested": secondary_ratio,
        "secondary_ratio_actual": round(len(secondary_rows) / len(primary_rows), 6),
        "secondary_sample_ids_sha256": stable_hash("\n".join(secondary_ids)),
        "blinding": "teacher_reference_separate_review_only",
        "label_status": "annotation_pending_not_gold",
        "files": {
            "primary": {"path": str(primary_file), "sha256": file_sha256(primary_file)},
            "secondary": {"path": str(secondary_file), "sha256": file_sha256(secondary_file)},
            "teacher_review_only": {"path": str(teacher_file), "sha256": file_sha256(teacher_file)},
        },
    }
    manifest_file = output_dir / "annotation_manifest.json"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    emit(**manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--secondary-ratio", type=float, default=0.20)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()

    if not args.input_file.is_file():
        emit("annotation_packet_complete", status="FAIL", error=f"File not found: {args.input_file}")
        return 1
    try:
        prepare_packet(args.input_file, args.output_dir, args.secondary_ratio, args.seed)
    except Exception as exc:
        emit("annotation_packet_complete", status="FAIL", error=repr(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
