"""Freeze a fully human-reviewed second-Schema CSV packet into immutable JSONL splits."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from evaluation.keyword_evaluator import file_sha256, read_jsonl, write_json_atomic, write_jsonl_atomic
from scripts.build_second_schema_assets import SPLIT_COUNTS, validate_rows


TRUE_VALUES = {"true", "yes", "y", "1", "通过"}


def read_review_packet(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "sample_id", "group_id", "split", "source", "intent", "urgency", "evidence",
        "challenge_slice", "approved", "correction_notes",
    }
    if not rows:
        raise ValueError("review packet is empty")
    if set(rows[0]) != required:
        raise ValueError(f"review packet columns must be exactly {sorted(required)}")
    converted = []
    for row_number, row in enumerate(rows, 2):
        if row["approved"].strip().lower() not in TRUE_VALUES:
            raise ValueError(f"row {row_number} ({row['sample_id']}) is not approved")
        converted.append({
            "sample_id": row["sample_id"].strip(),
            "group_id": row["group_id"].strip(),
            "split": row["split"].strip(),
            "source": row["source"].strip(),
            "target": {
                "intent": row["intent"].strip(),
                "urgency": row["urgency"].strip(),
                "evidence": row["evidence"].strip(),
            },
            "challenge_slice": row["challenge_slice"].strip() or None,
            "synthetic": True,
            "human_verified": False,
            "annotation_status": "draft",
        })
    return converted


def _validate_draft_integrity(rows: list[dict[str, Any]], draft_dir: Path) -> str:
    manifest_path = draft_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "DRAFT_REQUIRES_HUMAN_REVIEW":
        raise ValueError("draft manifest has an unexpected status")
    originals = []
    for split in SPLIT_COUNTS:
        path = draft_dir / f"{split}.jsonl"
        if file_sha256(path) != manifest["files"][split]["sha256"]:
            raise ValueError(f"{split} draft SHA256 differs from its manifest")
        originals.extend(read_jsonl(path))
    original_by_id = {row["sample_id"]: row for row in originals}
    if set(original_by_id) != {row["sample_id"] for row in rows}:
        raise ValueError("review packet sample IDs differ from the immutable draft")
    immutable_fields = ("group_id", "split", "source", "challenge_slice")
    for row in rows:
        original = original_by_id[row["sample_id"]]
        for field in immutable_fields:
            if row[field] != original[field]:
                raise ValueError(f"{row['sample_id']} changed immutable field: {field}")
    return file_sha256(manifest_path)


def run(
    review_packet: Path,
    output_dir: Path,
    reviewer: str,
    review_date: str,
    draft_dir: Path | None = None,
) -> dict[str, Any]:
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError("reviewer must be non-empty")
    if len(review_date) != 10 or review_date[4] != "-" or review_date[7] != "-":
        raise ValueError("review-date must use YYYY-MM-DD")
    rows = read_review_packet(review_packet)
    draft_manifest_sha256 = _validate_draft_integrity(rows, draft_dir or review_packet.parent)
    validation = validate_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for row in rows:
        row["human_verified"] = True
        row["annotation_status"] = "frozen"
        row["reviewer"] = reviewer
        row["review_date"] = review_date
    for split, expected_count in SPLIT_COUNTS.items():
        path = output_dir / f"{split}.jsonl"
        write_jsonl_atomic(path, [row for row in rows if row["split"] == split])
        files[split] = {"path": str(path.as_posix()), "sha256": file_sha256(path), "rows": expected_count}
    manifest = {
        "report_version": "intent-routing-frozen-v1",
        "status": "FROZEN_FOR_ONE_SHOT_SMOKE_TEST",
        "synthetic": True,
        "human_verified": True,
        "frozen": True,
        "reviewer": reviewer,
        "review_date": review_date,
        "draft_manifest_sha256": draft_manifest_sha256,
        "source_review_packet": {"path": str(review_packet.as_posix()), "sha256": file_sha256(review_packet)},
        "validation": validation,
        "files": files,
        "protocol": {
            "dev_use": "prompt_contract_check_only",
            "test_use": "one_shot_final_comparison",
            "challenge_use": "diagnostic_only",
            "prohibited": ["training", "DPO", "GRPO", "prompt_sweep", "E4_adapter"],
        },
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-packet", type=Path, default=Path("data/derived/intent_routing_v1_draft/review_packet.csv"))
    parser.add_argument("--draft-dir", type=Path, default=Path("data/derived/intent_routing_v1_draft"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/frozen/intent_routing_v1"))
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--review-date", required=True)
    args = parser.parse_args()
    try:
        result = run(args.review_packet, args.output_dir, args.reviewer, args.review_date, args.draft_dir)
        print(json.dumps({"event": "second_schema_freeze_complete", **result}, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"event": "second_schema_freeze_complete", "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
