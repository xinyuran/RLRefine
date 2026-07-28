"""Create a tiny train-only GRPO dataset for a real ms-swift reward probe."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.split_canonical_data import file_sha256


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def prepare(input_file: Path, output_file: Path, sample_count: int) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    with input_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            messages = row.get("messages")
            solution = row.get("solution")
            if not isinstance(messages, list) or not messages or not solution:
                continue
            prompt_chars = sum(len(str(message.get("content", ""))) for message in messages if isinstance(message, dict))
            candidates.append({
                "sample_id": row.get("sample_id"),
                "prompt_chars": prompt_chars,
                "row": {"messages": messages, "solution": solution},
            })

    selected = sorted(candidates, key=lambda item: (item["prompt_chars"], str(item["sample_id"])))[:sample_count]
    if len(selected) != sample_count:
        raise ValueError(f"Expected {sample_count} valid rows, found {len(selected)}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item["row"], ensure_ascii=False) + "\n")
    os.replace(temporary, output_file)

    result = {
        "status": "PASS",
        "source_file": str(input_file),
        "source_sha256": file_sha256(input_file),
        "output_file": str(output_file),
        "output_sha256": file_sha256(output_file),
        "rows": len(selected),
        "rows_with_solution": sum(bool(item["row"]["solution"]) for item in selected),
        "message_roles": [
            [message.get("role") for message in item["row"]["messages"]]
            for item in selected
        ],
        "prompt_char_counts": [item["prompt_chars"] for item in selected],
        "sample_ids": [item["sample_id"] for item in selected],
    }
    emit("ms_swift_dry_run_data_complete", **result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=2)
    args = parser.parse_args()
    try:
        prepare(args.input_file, args.output_file, args.sample_count)
    except Exception as exc:
        emit("ms_swift_dry_run_data_complete", status="FAIL", error_type=type(exc).__name__, error=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
