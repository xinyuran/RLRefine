"""Validate completed blind annotations and prepare double-annotation review."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_annotation_packet import (
    ANNOTATION_COLUMNS,
    DEFAULT_SEED,
    prepare_rows,
    read_jsonl,
)
from scripts.split_canonical_data import file_sha256


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def read_annotation_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ANNOTATION_COLUMNS:
            raise ValueError(
                f"Unexpected columns in {path}: {reader.fieldnames}; "
                f"expected {list(ANNOTATION_COLUMNS)}"
            )
        return list(reader)


def issue(
    file_label: str,
    row_number: int,
    sample_id: str,
    field: str,
    code: str,
    message: str,
) -> Dict[str, Any]:
    return {
        "file": file_label,
        "row_number": row_number,
        "sample_id": sample_id,
        "field": field,
        "code": code,
        "message": message,
    }


def is_chinese_keyword(value: str) -> bool:
    return 1 <= len(value) <= 4 and all("\u4e00" <= char <= "\u9fff" for char in value)


def validate_rows(
    rows: Sequence[Dict[str, str]],
    expected_rows: Sequence[Dict[str, str]],
    file_label: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    normalized: Dict[str, Dict[str, Any]] = {}
    expected_by_id = {row["sample_id"]: row for row in expected_rows}
    seen_ids = set()

    if len(rows) != len(expected_rows):
        issues.append(
            issue(
                file_label,
                0,
                "",
                "sample_id",
                "row_count_mismatch",
                f"Expected {len(expected_rows)} rows, found {len(rows)}",
            )
        )

    for row_number, row in enumerate(rows, 2):
        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id:
            issues.append(issue(file_label, row_number, "", "sample_id", "missing", "sample_id is required"))
            continue
        if sample_id in seen_ids:
            issues.append(issue(file_label, row_number, sample_id, "sample_id", "duplicate", "Duplicate sample_id"))
            continue
        seen_ids.add(sample_id)

        expected = expected_by_id.get(sample_id)
        if expected is None:
            issues.append(issue(file_label, row_number, sample_id, "sample_id", "unexpected", "Unexpected sample_id"))
            continue
        source_text = row.get("source_text") or ""
        if source_text != expected["source_text"]:
            issues.append(
                issue(file_label, row_number, sample_id, "source_text", "modified", "source_text differs from frozen packet")
            )

        annotator_id = (row.get("annotator_id") or "").strip()
        status = (row.get("annotation_status") or "").strip()
        keywords_raw = (row.get("keywords_json") or "").strip()
        exclude_reason = (row.get("exclude_reason") or "").strip()
        notes = (row.get("notes") or "").strip()

        if not annotator_id:
            issues.append(issue(file_label, row_number, sample_id, "annotator_id", "missing", "annotator_id is required"))
        if status not in {"complete", "exclude"}:
            issues.append(
                issue(file_label, row_number, sample_id, "annotation_status", "invalid", "Status must be complete or exclude")
            )
            continue

        keywords: List[str] = []
        if status == "exclude":
            if not exclude_reason:
                issues.append(
                    issue(file_label, row_number, sample_id, "exclude_reason", "missing", "exclude requires a reason")
                )
            if keywords_raw not in {"", "[]"}:
                issues.append(
                    issue(file_label, row_number, sample_id, "keywords_json", "not_empty_for_exclude", "Excluded rows must not contain keywords")
                )
        else:
            if exclude_reason:
                issues.append(
                    issue(file_label, row_number, sample_id, "exclude_reason", "unexpected", "Complete rows must not have exclude_reason")
                )
            try:
                parsed = json.loads(keywords_raw)
            except json.JSONDecodeError as exc:
                issues.append(
                    issue(file_label, row_number, sample_id, "keywords_json", "invalid_json", str(exc))
                )
                parsed = None
            if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed or []):
                issues.append(
                    issue(file_label, row_number, sample_id, "keywords_json", "invalid_type", "Must be a JSON array of strings")
                )
            else:
                keywords = parsed
                if len(keywords) > 15:
                    issues.append(issue(file_label, row_number, sample_id, "keywords_json", "too_many", "At most 15 keywords"))
                if len(keywords) != len(set(keywords)):
                    issues.append(issue(file_label, row_number, sample_id, "keywords_json", "duplicate", "Keywords must be unique"))
                if not keywords and not notes:
                    issues.append(
                        issue(file_label, row_number, sample_id, "notes", "missing_for_empty", "Empty keyword list requires notes")
                    )
                for keyword in keywords:
                    if keyword != keyword.strip():
                        issues.append(issue(file_label, row_number, sample_id, "keywords_json", "whitespace", f"Keyword has outer whitespace: {keyword!r}"))
                    if not is_chinese_keyword(keyword):
                        issues.append(issue(file_label, row_number, sample_id, "keywords_json", "invalid_keyword", f"Keyword must contain 1-4 Chinese characters: {keyword!r}"))
                    if keyword not in source_text:
                        issues.append(issue(file_label, row_number, sample_id, "keywords_json", "not_in_source", f"Keyword is not an exact source substring: {keyword!r}"))

        normalized[sample_id] = {
            "sample_id": sample_id,
            "source_text": source_text,
            "annotator_id": annotator_id,
            "status": status,
            "keywords": keywords,
            "exclude_reason": exclude_reason,
            "notes": notes,
        }

    missing_ids = sorted(set(expected_by_id) - seen_ids)
    for sample_id in missing_ids:
        issues.append(issue(file_label, 0, sample_id, "sample_id", "missing_expected", "Expected sample_id is absent"))
    return issues, normalized


def keyword_f1(first: Sequence[str], second: Sequence[str]) -> float:
    first_set, second_set = set(first), set(second)
    if not first_set and not second_set:
        return 1.0
    if not first_set or not second_set:
        return 0.0
    overlap = len(first_set & second_set)
    return 2 * overlap / (len(first_set) + len(second_set))


def calculate_agreement(
    primary: Dict[str, Dict[str, Any]],
    secondary: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    status_agree = 0
    complete_pairs = 0
    exact_complete = 0
    f1_values: List[float] = []
    auto_agreed = 0
    disagreements: List[Dict[str, str]] = []

    for sample_id in sorted(secondary):
        first, second = primary[sample_id], secondary[sample_id]
        same_status = first["status"] == second["status"]
        status_agree += int(same_status)
        agreed = False
        if first["status"] == second["status"] == "exclude":
            agreed = True
        elif first["status"] == second["status"] == "complete":
            complete_pairs += 1
            score = keyword_f1(first["keywords"], second["keywords"])
            f1_values.append(score)
            exact = set(first["keywords"]) == set(second["keywords"])
            exact_complete += int(exact)
            agreed = exact
        if agreed:
            auto_agreed += 1
            continue
        disagreements.append(
            {
                "sample_id": sample_id,
                "source_text": first["source_text"],
                "primary_annotator_id": first["annotator_id"],
                "primary_status": first["status"],
                "primary_keywords_json": json.dumps(first["keywords"], ensure_ascii=False),
                "primary_exclude_reason": first["exclude_reason"],
                "primary_notes": first["notes"],
                "secondary_annotator_id": second["annotator_id"],
                "secondary_status": second["status"],
                "secondary_keywords_json": json.dumps(second["keywords"], ensure_ascii=False),
                "secondary_exclude_reason": second["exclude_reason"],
                "secondary_notes": second["notes"],
                "adjudicator_id": "",
                "adjudicated_status": "",
                "adjudicated_keywords_json": "",
                "adjudication_reason": "",
                "adjudication_notes": "",
            }
        )

    total = len(secondary)
    metrics = {
        "double_annotated_rows": total,
        "status_agreement_count": status_agree,
        "status_agreement_rate": round(status_agree / total, 6) if total else 0.0,
        "complete_pair_count": complete_pairs,
        "exact_keyword_set_count": exact_complete,
        "exact_keyword_set_rate_among_complete_pairs": round(exact_complete / complete_pairs, 6) if complete_pairs else None,
        "mean_keyword_set_f1_among_complete_pairs": round(sum(f1_values) / len(f1_values), 6) if f1_values else None,
        "auto_agreed_rows": auto_agreed,
        "disagreement_rows": len(disagreements),
    }
    return metrics, disagreements


def write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_disagreements(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "sample_id", "source_text", "primary_annotator_id", "primary_status",
        "primary_keywords_json", "primary_exclude_reason", "primary_notes",
        "secondary_annotator_id", "secondary_status", "secondary_keywords_json",
        "secondary_exclude_reason", "secondary_notes", "adjudicator_id",
        "adjudicated_status", "adjudicated_keywords_json", "adjudication_reason",
        "adjudication_notes",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def validate_annotations(
    primary_file: Path,
    secondary_file: Path,
    test_candidate_file: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    emit(
        "annotation_validation_inputs",
        primary={"path": str(primary_file.resolve()), "exists": primary_file.is_file()},
        secondary={"path": str(secondary_file.resolve()), "exists": secondary_file.is_file()},
        test_candidate={"path": str(test_candidate_file.resolve()), "exists": test_candidate_file.is_file()},
        output_dir=str(output_dir.resolve()),
    )
    records = read_jsonl(test_candidate_file)
    expected_primary, expected_secondary, _ = prepare_rows(records, secondary_ratio=0.20, seed=DEFAULT_SEED)
    primary_rows = read_annotation_csv(primary_file)
    secondary_rows = read_annotation_csv(secondary_file)
    primary_issues, primary = validate_rows(primary_rows, expected_primary, "primary")
    secondary_issues, secondary = validate_rows(secondary_rows, expected_secondary, "secondary")
    issues = primary_issues + secondary_issues

    primary_annotators = sorted({row["annotator_id"] for row in primary.values() if row["annotator_id"]})
    secondary_annotators = sorted({row["annotator_id"] for row in secondary.values() if row["annotator_id"]})
    if len(primary_annotators) != 1:
        issues.append(issue("primary", 0, "", "annotator_id", "not_stable", f"Expected one stable ID, found {primary_annotators}"))
    if len(secondary_annotators) != 1:
        issues.append(issue("secondary", 0, "", "annotator_id", "not_stable", f"Expected one stable ID, found {secondary_annotators}"))
    if set(primary_annotators) & set(secondary_annotators):
        issues.append(issue("both", 0, "", "annotator_id", "not_independent", "Primary and secondary annotator IDs must differ"))

    report: Dict[str, Any] = {
        "event": "annotation_validation_complete",
        "status": "FAIL" if issues else "PASS",
        "label_status": "annotation_invalid_not_gold" if issues else "annotation_valid_pending_adjudication_not_gold",
        "inputs": {
            "primary": {"path": str(primary_file), "sha256": file_sha256(primary_file), "rows": len(primary_rows)},
            "secondary": {"path": str(secondary_file), "sha256": file_sha256(secondary_file), "rows": len(secondary_rows)},
            "test_candidate": {"path": str(test_candidate_file), "sha256": file_sha256(test_candidate_file)},
        },
        "annotators": {"primary": primary_annotators, "secondary": secondary_annotators},
        "validation_issue_count": len(issues),
        "validation_issue_counts": {},
        "validation_issues": issues[:100],
    }
    for item in issues:
        key = f"{item['file']}:{item['code']}"
        report["validation_issue_counts"][key] = report["validation_issue_counts"].get(key, 0) + 1

    disagreement_file = output_dir / "annotation_disagreements.csv"
    if not issues:
        metrics, disagreements = calculate_agreement(primary, secondary)
        report["agreement"] = metrics
        write_disagreements(disagreement_file, disagreements)
        report["disagreement_file"] = {
            "path": str(disagreement_file),
            "sha256": file_sha256(disagreement_file),
        }
    write_json_atomic(output_dir / "annotation_validation_report.json", report)
    emit(**report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-file", type=Path, required=True)
    parser.add_argument("--secondary-file", type=Path, required=True)
    parser.add_argument("--test-candidate-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = validate_annotations(
            args.primary_file,
            args.secondary_file,
            args.test_candidate_file,
            args.output_dir,
        )
    except Exception as exc:
        emit(
            "annotation_validation_complete",
            status="FAIL",
            error_type=type(exc).__name__,
            error=str(exc),
            filename=getattr(exc, "filename", None),
        )
        return 1
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
