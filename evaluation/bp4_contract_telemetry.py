"""Contract-aware telemetry for raw BP4 model completions.

The standard evaluator intentionally treats contract-invalid predictions as empty.
That is correct for F1, but it hides hallucinated keywords inside parseable JSON
that failed another target-contract rule. This module measures those raw proposals
without relaxing the target contract.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, Mapping, Sequence


def _raw_keywords(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    items = payload.get("keywords")
    if not isinstance(items, list):
        return []
    values = []
    for item in items:
        if isinstance(item, list) and len(item) >= 2 and isinstance(item[1], str):
            values.append(item[1])
    return values


def contract_aware_telemetry(
    predictions: Sequence[Mapping[str, Any]],
    source_by_id: Mapping[str, str],
) -> Dict[str, Any]:
    """Measure raw keyword faithfulness even when target validation failed."""
    categories: Counter[str] = Counter()
    parseable = 0
    proposed = 0
    hallucinated = 0
    hallucinated_samples = 0
    samples = []
    for row in predictions:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in source_by_id:
            raise ValueError("Telemetry prediction has an unknown sample_id")
        raw = row.get("raw_response")
        raw = raw if isinstance(raw, str) else ""
        try:
            payload = json.loads(raw.strip())
            parseable += 1
        except json.JSONDecodeError:
            categories["exact_json_parse_failed"] += 1
            continue
        keywords = _raw_keywords(payload)
        missing = [word for word in keywords if word not in source_by_id[sample_id]]
        proposed += len(keywords)
        hallucinated += len(missing)
        if missing:
            hallucinated_samples += 1
        for error in row.get("validation_errors") or []:
            category = str(error).split(".", 1)[-1]
            categories[category] += 1
        if row.get("status") != "success" and len(samples) < 20:
            samples.append(
                {
                    "sample_id": sample_id,
                    "error_code": row.get("error_code"),
                    "validation_errors": row.get("validation_errors") or [],
                    "raw_keyword_count": len(keywords),
                    "hallucinated_keywords": missing,
                }
            )
    count = len(predictions)
    return {
        "telemetry_version": "bp4-contract-aware-telemetry-v1",
        "sample_count": count,
        "exact_json_parseable_count": parseable,
        "exact_json_parseable_rate": parseable / count if count else 0.0,
        "raw_keyword_count": proposed,
        "raw_hallucinated_keyword_count": hallucinated,
        "raw_hallucination_keyword_rate": (
            hallucinated / proposed if proposed else 0.0
        ),
        "raw_hallucinated_sample_count": hallucinated_samples,
        "raw_hallucinated_sample_rate": (
            hallucinated_samples / count if count else 0.0
        ),
        "contract_error_counts": dict(sorted(categories.items())),
        "invalid_samples": samples,
        "interpretation": (
            "Counts keyword strings recoverable from exact JSON, including "
            "target-contract-invalid outputs; malformed JSON has no recoverable "
            "keyword denominator."
        ),
    }
