"""Stage-neutral helpers for BP4 post-training evaluation."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def select_best_checkpoint(root: Path, artifact_path: Path) -> Path:
    report = json.loads(artifact_path.read_text(encoding="utf-8"))
    value = report.get("best_model_checkpoint")
    if not isinstance(value, str) or not value:
        raise ValueError("Training artifacts do not identify best_model_checkpoint")
    checkpoint = Path(value)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    if not checkpoint.is_dir():
        raise ValueError(f"Best candidate checkpoint does not exist: {checkpoint}")
    if not (checkpoint / "trainer_state.json").is_file():
        raise ValueError("Best candidate checkpoint lacks trainer_state.json")
    if not (checkpoint / "adapter_config.json").is_file():
        raise ValueError("Best candidate checkpoint lacks adapter_config.json")
    return checkpoint


def metric_summary(
    report: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    telemetry: Mapping[str, Any],
    max_new_tokens: int,
) -> Dict[str, Any]:
    resources = report["resource_metrics"]
    return {
        "micro_f1": report["micro"]["f1"],
        "macro_f1": report["macro"]["f1"],
        "schema_valid_rate": report["schema_valid_rate"],
        "hallucination_keyword_rate": telemetry[
            "raw_hallucination_keyword_rate"
        ],
        "hallucination_metric_version": telemetry["telemetry_version"],
        "standard_valid_only_hallucination_keyword_rate": report[
            "hallucination"
        ]["keyword_rate"],
        "mean_input_tokens": resources["mean_input_tokens"],
        "mean_output_tokens": resources["mean_output_tokens"],
        "mean_request_seconds": resources["mean_request_seconds"],
        "max_token_output_count": sum(
            row.get("output_tokens", 0) >= max_new_tokens
            for row in predictions
        ),
        "error_sample_count": report["error_sample_count"],
    }


def sample_f1(
    references: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> list[float]:
    by_id = {row["sample_id"]: row for row in predictions}
    values = []
    for reference in references:
        prediction = by_id[reference["sample_id"]]
        data = prediction.get("data") if prediction.get("status") != "error" else None
        predicted = (
            {item[1] for item in data["keywords"]}
            if isinstance(data, dict) and isinstance(data.get("keywords"), list)
            else set()
        )
        gold = set(reference["gold"]["keywords"])
        tp = len(predicted & gold)
        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(gold) if gold else 0.0
        values.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return values


def paired_bootstrap(
    baseline_scores: Sequence[float],
    candidate_scores: Sequence[float],
    *,
    iterations: int,
    seed: int,
) -> Dict[str, Any]:
    if len(baseline_scores) != len(candidate_scores) or not baseline_scores:
        raise ValueError("Paired bootstrap requires non-empty aligned scores")
    deltas = [
        candidate - baseline
        for baseline, candidate in zip(baseline_scores, candidate_scores)
    ]
    rng = random.Random(seed)
    count = len(deltas)
    samples = sorted(
        sum(deltas[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(iterations)
    )

    def percentile(probability: float) -> float:
        position = probability * (len(samples) - 1)
        lower = int(position)
        upper = min(lower + 1, len(samples) - 1)
        fraction = position - lower
        return samples[lower] * (1 - fraction) + samples[upper] * fraction

    return {
        "macro_f1_delta_candidate_minus_baseline": sum(deltas) / count,
        "macro_f1_bootstrap_95_ci": [percentile(0.025), percentile(0.975)],
        "candidate_wins": sum(value > 0 for value in deltas),
        "baseline_wins": sum(value < 0 for value in deltas),
        "ties": sum(value == 0 for value in deltas),
    }
