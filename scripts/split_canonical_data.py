"""Create deterministic group-level train/dev/test candidate splits."""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


SPLIT_VERSION = "keyword-v1-split-v1"
DEFAULT_SEED = "structalign-keyword-v1"
SPLIT_NAMES = ("train", "dev", "test_candidate")


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_ratios(train_ratio: float, dev_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, dev_ratio, test_ratio)
    if any(value <= 0 or value >= 1 for value in ratios):
        raise ValueError("Each split ratio must be between 0 and 1")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")


def split_records(
    records: Sequence[Dict[str, Any]],
    train_ratio: float = 0.75,
    dev_ratio: float = 0.10,
    test_ratio: float = 0.15,
    seed: str = DEFAULT_SEED,
) -> Dict[str, List[Dict[str, Any]]]:
    validate_ratios(train_ratio, dev_ratio, test_ratio)
    if not records:
        raise ValueError("Cannot split an empty dataset")

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_sample_ids = set()
    for record in records:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Every canonical record must have a non-empty sample_id")
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)
        group_id = record.get("group_id") or sample_id
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"Invalid group_id for sample {sample_id}")
        groups[group_id].append(record)

    ordered_group_ids = sorted(
        groups,
        key=lambda group_id: stable_hash(f"{seed}:{group_id}"),
    )
    group_count = len(ordered_group_ids)
    test_group_count = round(group_count * test_ratio)
    dev_group_count = round(group_count * dev_ratio)

    group_assignment = {}
    for index, group_id in enumerate(ordered_group_ids):
        if index < test_group_count:
            split_name = "test_candidate"
        elif index < test_group_count + dev_group_count:
            split_name = "dev"
        else:
            split_name = "train"
        group_assignment[group_id] = split_name

    result = {name: [] for name in SPLIT_NAMES}
    for group_id in ordered_group_ids:
        split_name = group_assignment[group_id]
        for record in groups[group_id]:
            output_record = dict(record)
            output_record["group_id"] = group_id
            output_record["split"] = split_name
            output_record["split_version"] = SPLIT_VERSION
            result[split_name].append(output_record)

    return result


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


def write_jsonl_atomic(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def verify_disjoint(splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    sample_sets = {
        name: {record["sample_id"] for record in records}
        for name, records in splits.items()
    }
    group_sets = {
        name: {record["group_id"] for record in records}
        for name, records in splits.items()
    }
    overlaps = {}
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1:]:
            overlaps[f"sample_{left}_{right}"] = len(sample_sets[left] & sample_sets[right])
            overlaps[f"group_{left}_{right}"] = len(group_sets[left] & group_sets[right])
    return overlaps


def split_file(
    input_file: Path,
    output_dir: Path,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: str,
) -> Dict[str, Any]:
    emit(
        "split_start",
        input_file=str(input_file),
        output_dir=str(output_dir),
        seed=seed,
        ratios={"train": train_ratio, "dev": dev_ratio, "test_candidate": test_ratio},
    )
    records = read_jsonl(input_file)
    splits = split_records(records, train_ratio, dev_ratio, test_ratio, seed)
    overlaps = verify_disjoint(splits)
    if any(overlaps.values()):
        raise ValueError(f"Cross-split leakage detected: {overlaps}")

    output_files = {}
    for split_name, split_records_list in splits.items():
        output_path = output_dir / f"sft_{split_name}.jsonl"
        write_jsonl_atomic(output_path, split_records_list)
        output_files[split_name] = output_path

    manifest = {
        "event": "split_complete",
        "split_version": SPLIT_VERSION,
        "seed": seed,
        "source_file": str(input_file),
        "source_sha256": file_sha256(input_file),
        "ratios": {"train": train_ratio, "dev": dev_ratio, "test_candidate": test_ratio},
        "input_rows": len(records),
        "counts": {name: len(items) for name, items in splits.items()},
        "group_counts": {
            name: len({record["group_id"] for record in items})
            for name, items in splits.items()
        },
        "overlaps": overlaps,
        "files": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
                "sample_ids_sha256": stable_hash(
                    "\n".join(sorted(record["sample_id"] for record in splits[name]))
                ),
            }
            for name, path in output_files.items()
        },
        "test_label_status": "teacher_generated_candidate_not_gold",
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    emit("overlap_check", **overlaps)
    emit(**manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--dev-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()

    if not args.input_file.is_file():
        emit("split_complete", status="FAIL", error=f"File not found: {args.input_file}")
        return 1
    try:
        split_file(
            args.input_file,
            args.output_dir,
            args.train_ratio,
            args.dev_ratio,
            args.test_ratio,
            args.seed,
        )
    except Exception as exc:
        emit("split_complete", status="FAIL", error=repr(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
