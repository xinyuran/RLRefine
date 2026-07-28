"""Validate adjudication results and freeze the human gold test set."""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_annotation_packet import DEFAULT_SEED, prepare_rows, read_jsonl
from scripts.split_canonical_data import file_sha256, stable_hash
from scripts.validate_annotations import (
    calculate_agreement,
    is_chinese_keyword,
    issue,
    read_annotation_csv,
    validate_rows,
    write_json_atomic,
)


GOLD_VERSION = "keyword-gold-v1"
ADJUDICATION_REASONS = {
    "primary_preferred",
    "secondary_preferred",
    "merged",
    "new_decision",
    "exclude",
}
ADJUDICATION_COLUMNS = (
    "sample_id",
    "source_text",
    "primary_annotator_id",
    "primary_status",
    "primary_keywords_json",
    "primary_exclude_reason",
    "primary_notes",
    "secondary_annotator_id",
    "secondary_status",
    "secondary_keywords_json",
    "secondary_exclude_reason",
    "secondary_notes",
    "adjudicator_id",
    "adjudicated_status",
    "adjudicated_keywords_json",
    "adjudication_reason",
    "adjudication_notes",
)
IMMUTABLE_ADJUDICATION_COLUMNS = ADJUDICATION_COLUMNS[:12]


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def read_adjudication_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ADJUDICATION_COLUMNS:
            raise ValueError(
                f"Unexpected columns in {path}: {reader.fieldnames}; "
                f"expected {list(ADJUDICATION_COLUMNS)}"
            )
        return list(reader)


def validate_adjudications(
    rows: Sequence[Dict[str, str]],
    expected_rows: Sequence[Dict[str, str]],
    forbidden_annotators: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    normalized: Dict[str, Dict[str, Any]] = {}
    expected_by_id = {row["sample_id"]: row for row in expected_rows}
    seen_ids = set()

    if len(rows) != len(expected_rows):
        issues.append(issue("adjudication", 0, "", "sample_id", "row_count_mismatch", f"Expected {len(expected_rows)} rows, found {len(rows)}"))

    for row_number, row in enumerate(rows, 2):
        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id:
            issues.append(issue("adjudication", row_number, "", "sample_id", "missing", "sample_id is required"))
            continue
        if sample_id in seen_ids:
            issues.append(issue("adjudication", row_number, sample_id, "sample_id", "duplicate", "Duplicate sample_id"))
            continue
        seen_ids.add(sample_id)
        expected = expected_by_id.get(sample_id)
        if expected is None:
            issues.append(issue("adjudication", row_number, sample_id, "sample_id", "unexpected", "Unexpected sample_id"))
            continue

        for column in IMMUTABLE_ADJUDICATION_COLUMNS:
            if (row.get(column) or "") != (expected.get(column) or ""):
                issues.append(issue("adjudication", row_number, sample_id, column, "modified", f"Frozen column differs: {column}"))

        adjudicator_id = (row.get("adjudicator_id") or "").strip()
        status = (row.get("adjudicated_status") or "").strip()
        keywords_raw = (row.get("adjudicated_keywords_json") or "").strip()
        reason = (row.get("adjudication_reason") or "").strip()
        notes = (row.get("adjudication_notes") or "").strip()

        if not adjudicator_id:
            issues.append(issue("adjudication", row_number, sample_id, "adjudicator_id", "missing", "adjudicator_id is required"))
        elif adjudicator_id in forbidden_annotators:
            issues.append(issue("adjudication", row_number, sample_id, "adjudicator_id", "not_independent", "Adjudicator must differ from primary and secondary annotators"))
        if status not in {"complete", "exclude"}:
            issues.append(issue("adjudication", row_number, sample_id, "adjudicated_status", "invalid", "Status must be complete or exclude"))
            continue
        if reason not in ADJUDICATION_REASONS:
            issues.append(issue("adjudication", row_number, sample_id, "adjudication_reason", "invalid", f"Unknown reason: {reason!r}"))
        if not notes:
            issues.append(issue("adjudication", row_number, sample_id, "adjudication_notes", "missing", "Adjudication notes are required"))

        try:
            parsed = json.loads(keywords_raw)
        except json.JSONDecodeError as exc:
            issues.append(issue("adjudication", row_number, sample_id, "adjudicated_keywords_json", "invalid_json", str(exc)))
            parsed = None
        keywords: List[str] = []
        if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed or []):
            issues.append(issue("adjudication", row_number, sample_id, "adjudicated_keywords_json", "invalid_type", "Must be a JSON array of strings"))
        else:
            keywords = parsed
            if len(keywords) > 15:
                issues.append(issue("adjudication", row_number, sample_id, "adjudicated_keywords_json", "too_many", "At most 15 keywords"))
            if len(keywords) != len(set(keywords)):
                issues.append(issue("adjudication", row_number, sample_id, "adjudicated_keywords_json", "duplicate", "Keywords must be unique"))
            if status == "complete" and not keywords and not notes:
                issues.append(issue("adjudication", row_number, sample_id, "adjudication_notes", "missing_for_empty", "Empty complete result requires notes"))
            if status == "exclude" and keywords:
                issues.append(issue("adjudication", row_number, sample_id, "adjudicated_keywords_json", "not_empty_for_exclude", "Excluded rows must have []"))
            for keyword in keywords:
                if keyword != keyword.strip():
                    issues.append(issue("adjudication", row_number, sample_id, "adjudicated_keywords_json", "whitespace", f"Keyword has outer whitespace: {keyword!r}"))
                if not is_chinese_keyword(keyword):
                    issues.append(issue("adjudication", row_number, sample_id, "adjudicated_keywords_json", "invalid_keyword", f"Keyword must contain 1-4 Chinese characters: {keyword!r}"))
                if keyword not in row["source_text"]:
                    issues.append(issue("adjudication", row_number, sample_id, "adjudicated_keywords_json", "not_in_source", f"Keyword is not an exact source substring: {keyword!r}"))

        if status == "exclude" and reason != "exclude":
            issues.append(issue("adjudication", row_number, sample_id, "adjudication_reason", "status_reason_mismatch", "Excluded rows require reason=exclude"))
        if status == "complete" and reason == "exclude":
            issues.append(issue("adjudication", row_number, sample_id, "adjudication_reason", "status_reason_mismatch", "Complete rows cannot use reason=exclude"))

        primary_keywords = json.loads(expected["primary_keywords_json"])
        secondary_keywords = json.loads(expected["secondary_keywords_json"])
        if reason == "primary_preferred" and (status != expected["primary_status"] or set(keywords) != set(primary_keywords)):
            issues.append(issue("adjudication", row_number, sample_id, "adjudicated_keywords_json", "reason_result_mismatch", "primary_preferred must match the primary label set"))
        if reason == "secondary_preferred" and (status != expected["secondary_status"] or set(keywords) != set(secondary_keywords)):
            issues.append(issue("adjudication", row_number, sample_id, "adjudicated_keywords_json", "reason_result_mismatch", "secondary_preferred must match the secondary label set"))

        normalized[sample_id] = {
            "sample_id": sample_id,
            "source_text": row["source_text"],
            "adjudicator_id": adjudicator_id,
            "status": status,
            "keywords": keywords,
            "reason": reason,
            "notes": notes,
        }

    for sample_id in sorted(set(expected_by_id) - seen_ids):
        issues.append(issue("adjudication", 0, sample_id, "sample_id", "missing_expected", "Expected disagreement is absent"))

    adjudicator_ids = sorted({row["adjudicator_id"] for row in normalized.values() if row["adjudicator_id"]})
    if len(adjudicator_ids) != 1:
        issues.append(issue("adjudication", 0, "", "adjudicator_id", "not_stable", f"Expected one stable ID, found {adjudicator_ids}"))
    return issues, normalized


def write_jsonl_atomic(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def build_gold_rows(
    candidate_records: Sequence[Dict[str, Any]],
    primary: Dict[str, Dict[str, Any]],
    secondary: Dict[str, Dict[str, Any]],
    adjudicated: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Counter]:
    record_by_id = {record["sample_id"]: record for record in candidate_records}
    gold_rows: List[Dict[str, Any]] = []
    excluded_rows: List[Dict[str, Any]] = []
    origin_counts: Counter = Counter()

    for sample_id in sorted(primary):
        first = primary[sample_id]
        second = secondary.get(sample_id)
        final = first
        label_origin = "human_primary_single"
        provenance = {"primary_annotator_id": first["annotator_id"]}
        if sample_id in adjudicated:
            final = adjudicated[sample_id]
            label_origin = "human_adjudicated"
            provenance.update({
                "secondary_annotator_id": second["annotator_id"],
                "adjudicator_id": final["adjudicator_id"],
                "adjudication_reason": final["reason"],
            })
        elif second is not None:
            same = (
                first["status"] == second["status"] == "exclude"
                or (
                    first["status"] == second["status"] == "complete"
                    and set(first["keywords"]) == set(second["keywords"])
                )
            )
            if not same:
                raise ValueError(f"Missing adjudication for disagreement: {sample_id}")
            label_origin = "human_double_agreed"
            provenance["secondary_annotator_id"] = second["annotator_id"]

        origin_counts[label_origin] += 1
        source_record = record_by_id[sample_id]
        common = {
            "sample_id": sample_id,
            "group_id": source_record.get("group_id") or sample_id,
            "source_line": source_record.get("source_line"),
            "raw_text": first["source_text"],
            "normalized_text": first["source_text"],
            "label_origin": label_origin,
            "annotation_provenance": provenance,
            "dataset_version": GOLD_VERSION,
        }
        if final["status"] == "exclude":
            excluded_rows.append({**common, "status": "exclude", "exclude_reason": final.get("notes", "")})
        else:
            gold_rows.append({**common, "gold": {"keywords": final["keywords"]}, "split": "test_gold"})
    return gold_rows, excluded_rows, origin_counts


def freeze_gold(
    primary_file: Path,
    secondary_file: Path,
    disagreement_file: Path,
    adjudicated_file: Path,
    test_candidate_file: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    emit(
        "gold_freeze_inputs",
        primary=str(primary_file.resolve()),
        secondary=str(secondary_file.resolve()),
        disagreements=str(disagreement_file.resolve()),
        adjudicated=str(adjudicated_file.resolve()),
        test_candidate=str(test_candidate_file.resolve()),
        output_dir=str(output_dir.resolve()),
    )
    candidate_records = read_jsonl(test_candidate_file)
    expected_primary, expected_secondary, _ = prepare_rows(candidate_records, secondary_ratio=0.20, seed=DEFAULT_SEED)
    primary_rows = read_annotation_csv(primary_file)
    secondary_rows = read_annotation_csv(secondary_file)
    primary_issues, primary = validate_rows(primary_rows, expected_primary, "primary")
    secondary_issues, secondary = validate_rows(secondary_rows, expected_secondary, "secondary")
    issues = primary_issues + secondary_issues

    _, generated_disagreements = calculate_agreement(primary, secondary)
    original_disagreements = read_adjudication_csv(disagreement_file)
    original_by_id = {row["sample_id"]: row for row in original_disagreements}
    generated_by_id = {row["sample_id"]: row for row in generated_disagreements}
    if len(original_by_id) != len(original_disagreements):
        issues.append(issue("disagreements", 0, "", "sample_id", "duplicate", "Duplicate sample_id in original disagreement file"))
    if set(original_by_id) != set(generated_by_id):
        issues.append(issue("disagreements", 0, "", "sample_id", "set_mismatch", "Original disagreement IDs differ from regenerated disagreements"))
    for sample_id in sorted(set(original_by_id) & set(generated_by_id)):
        for column in ADJUDICATION_COLUMNS:
            if (original_by_id[sample_id].get(column) or "") != (generated_by_id[sample_id].get(column) or ""):
                issues.append(issue("disagreements", 0, sample_id, column, "modified", f"Original disagreement column differs: {column}"))

    adjudicated_rows = read_adjudication_csv(adjudicated_file)
    forbidden = sorted({row["annotator_id"] for row in list(primary.values()) + list(secondary.values())})
    adjudication_issues, adjudicated = validate_adjudications(adjudicated_rows, generated_disagreements, forbidden)
    issues.extend(adjudication_issues)

    report: Dict[str, Any] = {
        "event": "gold_freeze_complete",
        "status": "FAIL" if issues else "PASS",
        "label_status": "adjudication_invalid_not_gold" if issues else "human_adjudicated_gold_frozen",
        "gold_version": GOLD_VERSION,
        "inputs": {
            "primary": {"path": str(primary_file), "sha256": file_sha256(primary_file), "rows": len(primary_rows)},
            "secondary": {"path": str(secondary_file), "sha256": file_sha256(secondary_file), "rows": len(secondary_rows)},
            "disagreements": {"path": str(disagreement_file), "sha256": file_sha256(disagreement_file), "rows": len(original_disagreements)},
            "adjudicated": {"path": str(adjudicated_file), "sha256": file_sha256(adjudicated_file), "rows": len(adjudicated_rows)},
            "test_candidate": {"path": str(test_candidate_file), "sha256": file_sha256(test_candidate_file), "rows": len(candidate_records)},
        },
        "validation_issue_count": len(issues),
        "validation_issue_counts": {},
        "validation_issues": issues[:100],
    }
    for item in issues:
        key = f"{item['file']}:{item['code']}"
        report["validation_issue_counts"][key] = report["validation_issue_counts"].get(key, 0) + 1

    if not issues:
        gold_rows, excluded_rows, origin_counts = build_gold_rows(candidate_records, primary, secondary, adjudicated)
        gold_file = output_dir / "gold_test.jsonl"
        excluded_file = output_dir / "gold_excluded.jsonl"
        write_jsonl_atomic(gold_file, gold_rows)
        write_jsonl_atomic(excluded_file, excluded_rows)
        reason_counts = Counter(row["reason"] for row in adjudicated.values())
        report.update({
            "counts": {
                "test_candidates": len(candidate_records),
                "gold_rows": len(gold_rows),
                "excluded_rows": len(excluded_rows),
                "primary_only": origin_counts["human_primary_single"],
                "double_agreed": origin_counts["human_double_agreed"],
                "adjudicated": origin_counts["human_adjudicated"],
            },
            "adjudicators": sorted({row["adjudicator_id"] for row in adjudicated.values()}),
            "adjudication_reason_counts": dict(sorted(reason_counts.items())),
            "outputs": {
                "gold_test": {"path": str(gold_file), "sha256": file_sha256(gold_file), "sample_ids_sha256": stable_hash("\n".join(sorted(row["sample_id"] for row in gold_rows)))},
                "excluded": {"path": str(excluded_file), "sha256": file_sha256(excluded_file)},
            },
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "gold_manifest.json"
    write_json_atomic(report_file, report)
    emit(**report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-file", type=Path, required=True)
    parser.add_argument("--secondary-file", type=Path, required=True)
    parser.add_argument("--disagreement-file", type=Path, required=True)
    parser.add_argument("--adjudicated-file", type=Path, required=True)
    parser.add_argument("--test-candidate-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = freeze_gold(
            args.primary_file,
            args.secondary_file,
            args.disagreement_file,
            args.adjudicated_file,
            args.test_candidate_file,
            args.output_dir,
        )
    except Exception as exc:
        emit("gold_freeze_complete", status="FAIL", error_type=type(exc).__name__, error=str(exc), filename=getattr(exc, "filename", None))
        return 1
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
