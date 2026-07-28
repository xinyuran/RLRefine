"""Independent acceptance gate and feedback packet for the complete BP1 asset package."""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from core.target_contract import (
    DATASET_VERSION,
    TARGET_CONTRACT_VERSION,
    build_messages,
    validate_keyword_payload,
)
from evaluation.keyword_evaluator import file_sha256
from scripts.build_bp1_assets import (
    BP1_VERSION,
    DEFAULT_GOLD_SHA256,
    NATURAL_SLICES,
    read_jsonl,
    source_and_payload,
    stable_hash,
    write_json_atomic,
)
from scripts.canonicalize_sft_data import extract_role_content, extract_source_text


EXPECTED_TRAIN_SHA256 = (
    "32064d8d2267df9472b7bbc9e1ad9d283b2e0c094b734694229c2d17c22bb522"
)
EXPECTED_DEV_SHA256 = (
    "1c7671805bbda658f5a6dad4bef5a35dd293f185747394a1c0c1ce85006501e8"
)


def emit(event: str, **payload: Any) -> None:
    print(
        json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


def validate_split_rows(
    rows: Sequence[Mapping[str, Any]],
    split: str,
) -> Dict[str, Any]:
    seen = set()
    group_ids = set()
    format_attacks = 0
    for row_number, row in enumerate(rows, 1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"{split} invalid/duplicate sample_id at row {row_number}")
        seen.add(sample_id)
        if row.get("split") != split:
            raise ValueError(f"{sample_id} split mismatch")
        if row.get("dataset_version") != DATASET_VERSION:
            raise ValueError(f"{sample_id} dataset version mismatch")
        if row.get("target_contract_version") != TARGET_CONTRACT_VERSION:
            raise ValueError(f"{sample_id} target contract mismatch")
        source, payload = source_and_payload(row)
        valid, errors = validate_keyword_payload(payload, source)
        if not valid:
            raise ValueError(f"{sample_id} payload invalid: {errors}")
        if row["messages"] != build_messages(source, payload):
            raise ValueError(f"{sample_id} prompt/target drift")
        assistant = extract_role_content(row["messages"], "assistant")
        if "<think>" in assistant or not assistant.startswith('{"keywords":'):
            raise ValueError(f"{sample_id} assistant is not JSON-only")
        group_ids.add(row.get("group_id") or sample_id)
        if "format_attack" in (row.get("challenge_slices") or []):
            format_attacks += 1
    return {
        "rows": len(rows),
        "sample_ids_sha256": stable_hash("\n".join(sorted(seen))),
        "group_count": len(group_ids),
        "format_attack_rows": format_attacks,
        "sample_ids": seen,
        "group_ids": group_ids,
    }


def build_gate_report(
    output_dir: Path,
    report_dir: Path,
    expected_train_sha256: str,
    expected_dev_sha256: str,
    expected_gold_sha256: str,
    expected_parent_train_rows: int = 2480,
    expected_dev_rows: int = 331,
) -> Dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: Dict[str, bool] = {
        "manifest_version_locked": manifest.get("manifest_version") == BP1_VERSION,
        "dataset_version_locked": manifest.get("dataset_version") == DATASET_VERSION,
        "target_contract_locked": (
            manifest.get("target_contract_version") == TARGET_CONTRACT_VERSION
        ),
        "parent_train_hash_locked": (
            manifest["source_files"]["train"]["sha256"] == expected_train_sha256
        ),
        "parent_dev_hash_locked": (
            manifest["source_files"]["dev"]["sha256"] == expected_dev_sha256
        ),
        "frozen_gold_hash_locked": (
            manifest["source_files"]["frozen_gold"]["sha256"]
            == expected_gold_sha256
        ),
        "frozen_gold_usage_is_overlap_only": (
            manifest["source_files"]["frozen_gold"]["usage"]
            == "sample_id_overlap_check_only_no_label_derivation"
        ),
        "frozen_gold_overlap_zero": manifest.get("frozen_gold_overlap") == 0,
        "all_recorded_overlaps_zero": not any(manifest.get("overlaps", {}).values()),
        "duplicate_repair_is_order_preserving": (
            manifest.get("safe_repair_policy", {}).get("duplicate_keywords")
            == "preserve_first_occurrence_in_existing_importance_order"
        ),
        "semantic_keyword_rewrite_forbidden": (
            manifest.get("safe_repair_policy", {}).get("semantic_keyword_rewrite")
            == "forbidden"
        ),
    }
    for name, item in manifest.get("source_files", {}).items():
        checks[f"source_{name}_file_hash_matches"] = (
            file_sha256(Path(item["path"])) == item["sha256"]
        )
    for name, item in manifest.get("documentation", {}).items():
        checks[f"documentation_{name}_hash_matches"] = (
            file_sha256(Path(item["path"])) == item["sha256"]
        )
    checks["all_required_documentation_present"] = set(
        manifest.get("documentation", {})
    ) == {
        "dataset_card",
        "target_contract",
        "target_contract_audit",
        "slice_manifest",
    }
    rows_by_split = {}
    validations = {}
    for split in ("train", "dev", "challenge"):
        path = output_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        rows_by_split[split] = rows
        validations[split] = validate_split_rows(rows, split)
        recorded = manifest["files"][split]
        checks[f"{split}_file_hash_matches"] = file_sha256(path) == recorded["sha256"]
        checks[f"{split}_sample_ids_hash_matches"] = (
            validations[split]["sample_ids_sha256"]
            == recorded["sample_ids_sha256"]
        )
        checks[f"{split}_count_matches"] = len(rows) == manifest["counts"][split]

    checks["dev_parent_count_preserved"] = (
        len(rows_by_split["dev"]) == expected_dev_rows
    )
    checks["parent_train_partition_complete"] = (
        len(rows_by_split["train"])
        + manifest["counts"]["challenge"]
        - validations["challenge"]["format_attack_rows"]
        == manifest["counts"]["parent_train"]
        == expected_parent_train_rows
    )
    checks["challenge_size_in_range"] = 100 <= len(rows_by_split["challenge"]) <= 200

    sample_overlap = {}
    group_overlap = {}
    names = list(rows_by_split)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            sample_overlap[f"{left}_{right}"] = len(
                validations[left]["sample_ids"] & validations[right]["sample_ids"]
            )
            group_overlap[f"{left}_{right}"] = len(
                validations[left]["group_ids"] & validations[right]["group_ids"]
            )
    checks["independent_sample_overlap_zero"] = not any(sample_overlap.values())
    checks["independent_group_overlap_zero"] = not any(group_overlap.values())

    slice_manifest_path = report_dir / "slice_manifest.json"
    slice_manifest = json.loads(slice_manifest_path.read_text(encoding="utf-8"))
    slice_counts = slice_manifest["slice_counts"]
    for slice_name in (*NATURAL_SLICES, "format_attack"):
        checks[f"slice_{slice_name}_present"] = slice_counts.get(slice_name, 0) > 0
    checks["format_attack_count_matches"] = (
        validations["challenge"]["format_attack_rows"]
        == slice_manifest["synthetic_format_attack_rows"]
        > 0
    )
    checks["review_queue_is_nonempty"] = (
        slice_manifest["review_queue_rows"] >= 50
    )
    checks["review_queue_is_not_gold"] = (
        slice_manifest["review_queue_policy"]
        == "diagnostic_only_not_training_or_evaluation_gold"
    )
    review_path = output_dir / "challenge_review_queue.jsonl"
    review_rows = read_jsonl(review_path)
    checks["review_queue_count_matches"] = (
        len(review_rows)
        == manifest["counts"]["challenge_review_queue"]
        == slice_manifest["review_queue_rows"]
    )
    checks["review_queue_file_hash_matches"] = (
        file_sha256(review_path)
        == manifest["files"]["challenge_review_queue"]["sha256"]
    )
    checks["review_queue_has_no_gold_field"] = all(
        "gold" not in row
        and row.get("label_status")
        == "needs_independent_human_review_not_evaluation_gold"
        for row in review_rows
    )

    target_audit = json.loads(
        (report_dir / "target_contract_audit.json").read_text(encoding="utf-8")
    )
    for split, rows in rows_by_split.items():
        audit = target_audit["by_split"][split]
        checks[f"{split}_all_targets_json_only"] = audit["json_only"] == len(rows)
        checks[f"{split}_all_prompts_match"] = (
            audit["prompt_contract_match"] == len(rows)
        )
        checks[f"{split}_all_payloads_valid"] = (
            audit["payload_valid"] == len(rows)
        )

    passed = all(checks.values())
    failed_checks = sorted(name for name, result in checks.items() if not result)
    return {
        "report_version": BP1_VERSION,
        "status": "PASS" if passed else "FAIL",
        "decision": "ACCEPT_BP1_ASSETS" if passed else "REJECT_BP1_ASSETS",
        "checks": checks,
        "failed_checks": failed_checks,
        "counts": manifest["counts"],
        "slice_counts": slice_counts,
        "sample_overlap": sample_overlap,
        "group_overlap": group_overlap,
        "evidence": {
            "manifest_file": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "target_audit_sha256": file_sha256(
                report_dir / "target_contract_audit.json"
            ),
            "slice_manifest_sha256": file_sha256(slice_manifest_path),
        },
    }


def build_feedback(report: Mapping[str, Any]) -> Dict[str, Any]:
    passed = report["status"] == "PASS"
    return {
        "packet_version": "bp1-feedback-v1",
        "decision": report["decision"],
        "next_work_package": (
            "BP2_POST_TRAINING_PIPELINE"
            if passed
            else "BP1_REPAIR_FROM_FAILED_CHECKS"
        ),
        "authorization": (
            "build_bp2_engineering; expensive_training_still_requires_stage_gate"
            if passed
            else "repair_bp1_only; no_training_or_gold_evaluation"
        ),
        "failed_checks": report["failed_checks"],
        "counts": report["counts"],
        "slice_counts": report["slice_counts"],
        "evidence": report["evidence"],
        "return_files": [
            "new_plan/logs/bp1_data_target_challenge.log",
            "data/canonical/keyword_v2/manifest.json",
            "reports/bp1_keyword_v2/bp1_gate.json",
            "reports/bp1_keyword_v2/bp1_feedback.json",
            "reports/bp1_keyword_v2/target_contract_audit.json",
            "reports/bp1_keyword_v2/slice_manifest.json",
        ],
    }


def render_markdown(report: Mapping[str, Any], feedback: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# BP1 Data / Target / Challenge Acceptance",
            "",
            f"- Status: `{report['status']}`",
            f"- Decision: `{report['decision']}`",
            f"- Next work package: `{feedback['next_work_package']}`",
            f"- Failed checks: {', '.join(report['failed_checks']) or 'none'}",
            f"- Counts: `{json.dumps(report['counts'], ensure_ascii=False)}`",
            f"- Slice counts: `{json.dumps(report['slice_counts'], ensure_ascii=False)}`",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/canonical/keyword_v2")
    )
    parser.add_argument(
        "--report-dir", type=Path, default=Path("reports/bp1_keyword_v2")
    )
    parser.add_argument("--expected-train-sha256", default=EXPECTED_TRAIN_SHA256)
    parser.add_argument("--expected-dev-sha256", default=EXPECTED_DEV_SHA256)
    parser.add_argument("--expected-gold-sha256", default=DEFAULT_GOLD_SHA256)
    parser.add_argument("--expected-parent-train-rows", type=int, default=2480)
    parser.add_argument("--expected-dev-rows", type=int, default=331)
    args = parser.parse_args()
    try:
        report = build_gate_report(
            args.output_dir,
            args.report_dir,
            args.expected_train_sha256,
            args.expected_dev_sha256,
            args.expected_gold_sha256,
            args.expected_parent_train_rows,
            args.expected_dev_rows,
        )
        feedback = build_feedback(report)
        write_json_atomic(args.report_dir / "bp1_gate.json", report)
        write_json_atomic(args.report_dir / "bp1_feedback.json", feedback)
        (args.report_dir / "bp1_summary.md").write_text(
            render_markdown(report, feedback), encoding="utf-8"
        )
        emit(
            "bp1_gate_complete",
            status=report["status"],
            decision=report["decision"],
            failed_checks=report["failed_checks"],
            next_work_package=feedback["next_work_package"],
            gate_file=str(args.report_dir / "bp1_gate.json"),
            gate_sha256=file_sha256(args.report_dir / "bp1_gate.json"),
            feedback_file=str(args.report_dir / "bp1_feedback.json"),
            feedback_sha256=file_sha256(args.report_dir / "bp1_feedback.json"),
        )
        return 0 if report["status"] == "PASS" else 1
    except Exception as exc:
        emit(
            "bp1_gate_complete",
            status="FAIL",
            decision="REJECT_BP1_ASSETS",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
