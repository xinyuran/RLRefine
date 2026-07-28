"""Validate GRPO reference wiring and reward components with JSONL logs.

This script is intended to run on the training server before GRPO starts. It
does not call a model or modify the dataset.
"""

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.reward_builder import RewardBuilder  # noqa: E402


LOGGER = logging.getLogger("p0_reward_validation")


def configure_logging(log_file: Path | None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers)


def emit(event: str, **payload: Any) -> None:
    LOGGER.info(json.dumps({"event": event, **payload}, ensure_ascii=False))


def read_jsonl(path: Path, sample_limit: int | None) -> Iterable[tuple[int, Optional[Dict[str, Any]]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if sample_limit is not None and line_number > sample_limit:
                break
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                emit("invalid_jsonl", line=line_number, error=str(exc))
                yield line_number, None


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(statistics.fmean(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def validate(path: Path, sample_limit: int | None) -> int:
    reward = RewardBuilder.create_keyword_reward()
    reward.config.enable_component_logging = False

    rows = 0
    missing_solution = 0
    invalid_jsonl_rows = 0
    invalid_message_rows = 0
    invalid_contract_rows = 0
    scoring_errors = 0
    details_rows: List[Dict[str, Any]] = []

    emit("validation_start", grpo_file=str(path), sample_limit=sample_limit)
    for line_number, row in read_jsonl(path, sample_limit):
        rows += 1
        if row is None:
            invalid_jsonl_rows += 1
            continue
        messages = row.get("messages")
        solution = row.get("solution")
        if not isinstance(messages, list) or not messages:
            invalid_message_rows += 1
            emit("invalid_messages", line=line_number)
            continue
        if not solution:
            missing_solution += 1
            if missing_solution <= 10:
                emit("missing_solution", line=line_number)
            continue

        try:
            # A reference scored against itself should have F1=1 when its data
            # contract is valid. This isolates reward/data wiring from the model.
            details = reward.score_with_details(solution, solution, messages)
            details_rows.append(details)
            if not details["valid_json"] or not details["schema_valid"]:
                invalid_contract_rows += 1
                if invalid_contract_rows <= 10:
                    emit(
                        "invalid_reference_contract",
                        line=line_number,
                        valid_json=details["valid_json"],
                        schema_valid=details["schema_valid"],
                        f1=details["f1"],
                    )
        except Exception as exc:  # log the exact server-side integration failure
            scoring_errors += 1
            if scoring_errors <= 10:
                emit("scoring_error", line=line_number, error=repr(exc))

    valid_json_count = sum(bool(item["valid_json"]) for item in details_rows)
    schema_valid_count = sum(bool(item["schema_valid"]) for item in details_rows)
    perfect_f1_count = sum(item["f1"] == 1.0 for item in details_rows)

    emit(
        "dataset_summary",
        rows=rows,
        scored_rows=len(details_rows),
        missing_solution=missing_solution,
        invalid_jsonl_rows=invalid_jsonl_rows,
        invalid_message_rows=invalid_message_rows,
        invalid_contract_rows=invalid_contract_rows,
        scoring_errors=scoring_errors,
        valid_json_rate=round(valid_json_count / len(details_rows), 6) if details_rows else 0.0,
        schema_valid_rate=round(schema_valid_count / len(details_rows), 6) if details_rows else 0.0,
        perfect_self_f1_rate=round(perfect_f1_count / len(details_rows), 6) if details_rows else 0.0,
    )

    for field_name in [
        "total", "thinking", "format", "quality", "f1",
        "accuracy", "hallucination_penalty",
    ]:
        emit(
            "component_summary",
            component=field_name,
            **summarize([float(item[field_name]) for item in details_rows]),
        )

    passed = (
        rows > 0
        and missing_solution == 0
        and invalid_jsonl_rows == 0
        and invalid_message_rows == 0
        and invalid_contract_rows == 0
        and scoring_errors == 0
        and len(details_rows) == rows
        and valid_json_count == rows
        and schema_valid_count == rows
        and perfect_f1_count == rows
    )
    emit("validation_complete", status="PASS" if passed else "FAIL")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grpo-file", type=Path, required=True)
    parser.add_argument("--sample-limit", type=int, default=200)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    configure_logging(args.log_file)
    if not args.grpo_file.is_file():
        emit("validation_complete", status="FAIL", error=f"File not found: {args.grpo_file}")
        return 1
    sample_limit = None if args.sample_limit <= 0 else args.sample_limit
    return validate(args.grpo_file, sample_limit)


if __name__ == "__main__":
    raise SystemExit(main())
