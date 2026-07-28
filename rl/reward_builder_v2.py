"""Reward V2 plugin for JSON-only keyword extraction GRPO."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

# This file is loaded by ms-swift as a standalone external plugin. Keep its
# runtime imports independent of the repository package root. Contract tests
# compare these frozen protocol values with core.keyword_contract.
MAX_KEYWORD_LENGTH = 4
MAX_KEYWORDS = 15

try:
    from swift.plugin import ORM, orms

    HAS_SWIFT = True
except ImportError:
    ORM = object
    orms = {}
    HAS_SWIFT = False


@dataclass(frozen=True)
class RewardV2Config:
    extraction_f1_weight: float = 0.65
    faithfulness_weight: float = 0.15
    atomicity_weight: float = 0.10
    count_calibration_weight: float = 0.10
    hallucination_penalty: float = 0.25
    invalid_json_reward: float = -1.0
    invalid_schema_reward: float = -0.75

    def __post_init__(self) -> None:
        total = (
            self.extraction_f1_weight
            + self.faithfulness_weight
            + self.atomicity_weight
            + self.count_calibration_weight
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Reward V2 positive weights must sum to 1.0, got {total}")
        if self.hallucination_penalty < 0:
            raise ValueError("hallucination_penalty must be non-negative")


def _keyword_items(payload: Mapping[str, Any]) -> List[List[Any]]:
    value = payload.get("keywords")
    return value if isinstance(value, list) else []


def _keyword_values(payload: Mapping[str, Any]) -> List[str]:
    return [
        str(item[1]).strip()
        for item in _keyword_items(payload)
        if isinstance(item, list) and len(item) == 3
    ]


def parse_exact_json(value: Any) -> Dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value.strip())
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def validate_structure(payload: Any) -> List[str]:
    if not isinstance(payload, dict) or set(payload) != {"keywords"}:
        return ["payload_top_level_invalid"]
    items = payload["keywords"]
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_KEYWORDS:
        return ["keyword_count_invalid"]
    errors: List[str] = []
    seen = set()
    for index, item in enumerate(items):
        prefix = f"keywords[{index}]"
        if not isinstance(item, list) or len(item) != 3:
            errors.append(f"{prefix}.shape_invalid")
            continue
        explanation, keyword, confidence = item
        if not isinstance(explanation, str) or not explanation.strip():
            errors.append(f"{prefix}.explanation_invalid")
        if not isinstance(keyword, str) or not 1 <= len(keyword) <= MAX_KEYWORD_LENGTH:
            errors.append(f"{prefix}.keyword_invalid")
        elif keyword in seen:
            errors.append(f"{prefix}.keyword_duplicate")
        else:
            seen.add(keyword)
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= float(confidence) <= 1
        ):
            errors.append(f"{prefix}.confidence_invalid")
    return errors


def extract_source_text(prompt: Any) -> str:
    if isinstance(prompt, list):
        user_contents = [
            str(item.get("content", ""))
            for item in prompt
            if isinstance(item, dict) and item.get("role") == "user"
        ]
        prompt = "\n".join(user_contents)
    elif isinstance(prompt, dict):
        prompt = str(prompt.get("content", ""))
    elif prompt is None:
        return ""
    else:
        prompt = str(prompt)
    marker = "【待处理评论】\n"
    if marker in prompt:
        text = prompt.split(marker, 1)[1]
        return text.split("\n\n请严格", 1)[0].strip()
    return prompt.strip()


def _f1(predicted: Sequence[str], reference: Sequence[str]) -> float:
    pred_set, ref_set = set(predicted), set(reference)
    if not pred_set or not ref_set:
        return 0.0
    overlap = len(pred_set & ref_set)
    precision = overlap / len(pred_set)
    recall = overlap / len(ref_set)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def _atomicity(predicted: Sequence[str], reference: Sequence[str]) -> float:
    if not predicted:
        return 0.0
    reference_set = set(reference)
    atomic = 0
    for value in predicted:
        if value in reference_set:
            atomic += 1
            continue
        contained = [item for item in reference if item and item in value]
        atomic += len(set(contained)) <= 1
    return atomic / len(predicted)


def _count_calibration(
    payload: Mapping[str, Any],
    predicted: Sequence[str],
    reference: Sequence[str],
) -> float:
    count_score = 1.0 - abs(len(predicted) - len(reference)) / max(
        len(predicted), len(reference), 1
    )
    calibration_terms = []
    reference_set = set(reference)
    for item in _keyword_items(payload):
        if not isinstance(item, list) or len(item) != 3:
            continue
        label = 1.0 if str(item[1]).strip() in reference_set else 0.0
        confidence = float(item[2])
        calibration_terms.append(1.0 - (confidence - label) ** 2)
    calibration = (
        sum(calibration_terms) / len(calibration_terms)
        if calibration_terms
        else 0.0
    )
    return (count_score + calibration) / 2


class SchemaBasedRewardV2(ORM if HAS_SWIFT else object):
    """Strict JSON reward with observable, independently ablatable components."""

    def __init__(self, config: RewardV2Config | None = None) -> None:
        self.config = config or RewardV2Config()
        self.logger = logging.getLogger(__name__)
        self._call_count = 0

    def score_with_details(
        self,
        completion: Any,
        solution: Any,
        prompt: Any,
    ) -> Dict[str, Any]:
        reference = parse_exact_json(solution)
        if reference is None or validate_structure(reference):
            raise ValueError("Reward V2 requires a structurally valid JSON solution")
        source = extract_source_text(prompt)
        payload = parse_exact_json(completion)
        details: Dict[str, Any] = {
            "valid_json": payload is not None,
            "schema_valid": False,
            "structural_errors": [],
            "extraction_f1": 0.0,
            "faithfulness": 0.0,
            "atomicity": 0.0,
            "count_calibration": 0.0,
            "hallucination_rate": 0.0,
            "hallucination_penalty": 0.0,
        }
        if payload is None:
            details["total"] = self.config.invalid_json_reward
            return details
        errors = validate_structure(payload)
        details["structural_errors"] = errors
        if errors:
            details["total"] = self.config.invalid_schema_reward
            return details
        details["schema_valid"] = True
        predicted = _keyword_values(payload)
        reference_values = _keyword_values(reference)
        faithful_count = sum(value in source for value in predicted) if source else 0
        details["faithfulness"] = faithful_count / len(predicted)
        details["hallucination_rate"] = 1.0 - details["faithfulness"]
        details["hallucination_penalty"] = (
            details["hallucination_rate"] * self.config.hallucination_penalty
        )
        details["extraction_f1"] = _f1(predicted, reference_values)
        details["atomicity"] = _atomicity(predicted, reference_values)
        details["count_calibration"] = _count_calibration(
            payload, predicted, reference_values
        )
        details["total"] = (
            self.config.extraction_f1_weight * details["extraction_f1"]
            + self.config.faithfulness_weight * details["faithfulness"]
            + self.config.atomicity_weight * details["atomicity"]
            + self.config.count_calibration_weight * details["count_calibration"]
            - details["hallucination_penalty"]
        )
        return details

    def __call__(
        self,
        completions: List[Any],
        solution: List[Any] | None = None,
        **kwargs: Any,
    ) -> List[float]:
        solutions = solution if solution is not None else kwargs.get("solutions")
        if solutions is None:
            raise ValueError("Reward V2 requires solution for every completion")
        prompts = kwargs.get("prompts")
        if prompts is None or (
            isinstance(prompts, list) and all(item is None for item in prompts)
        ):
            prompts = kwargs.get("messages")
        if not isinstance(solutions, (list, tuple)) or len(solutions) != len(completions):
            raise ValueError("solution batch size must match completions")
        if prompts is None:
            prompts = [None] * len(completions)
        if not isinstance(prompts, (list, tuple)) or len(prompts) != len(completions):
            raise ValueError("prompt batch size must match completions")
        details = [
            self.score_with_details(completion, reference, prompt)
            for completion, reference, prompt in zip(completions, solutions, prompts)
        ]
        self._call_count += 1
        summary = {
            "event": "reward_v2_component_summary",
            "call": self._call_count,
            "batch_size": len(details),
            "mean_total": sum(item["total"] for item in details) / len(details),
            "schema_valid_rate": sum(item["schema_valid"] for item in details)
            / len(details),
            "mean_hallucination_rate": sum(
                item["hallucination_rate"] for item in details
            )
            / len(details),
        }
        self.logger.info("REWARD_V2_METRICS %s", json.dumps(summary, sort_keys=True))
        return [float(item["total"]) for item in details]


# ms-swift treats entries in ``orms`` as factories and constructs them with
# framework-provided keyword arguments. Register the class rather than a
# pre-built callable instance so construction does not invoke ``__call__``.
schema_based_reward_v2 = SchemaBasedRewardV2
if HAS_SWIFT and isinstance(orms, dict):
    orms["schema_based_reward_v2"] = schema_based_reward_v2
