"""Build and independently gate BP3 Reward V2 and preference assets."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml

from core.target_contract import (
    DATASET_VERSION,
    TARGET_CONTRACT_VERSION,
    validate_keyword_payload,
)
from evaluation.keyword_evaluator import file_sha256
from rl.reward_builder_v2 import RewardV2Config, SchemaBasedRewardV2
from scripts.bp2_pipeline import _environment_lineage, _git_lineage, _source_lineage
from scripts.build_bp1_assets import read_jsonl, source_and_payload


BP3_VERSION = "bp3-reward-preferences-v1"
ATTACK_TYPES = (
    "invalid_json",
    "markdown_wrapper",
    "extra_top_level_field",
    "string_confidence",
    "miscalibrated_confidence",
    "hallucination",
    "duplicate_keyword",
    "empty_keywords",
    "drop_keyword",
    "merge_atomicity",
)


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True))


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def prompt_messages(row: Mapping[str, Any]) -> List[Dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"Invalid BP1 messages for {row.get('sample_id')}")
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"Missing assistant target for {row.get('sample_id')}")
    return copy.deepcopy(messages[:-1])


def _fake_keyword(source: str) -> str:
    for candidate in ("伪造词", "虚构词", "不存在"):
        if candidate not in source:
            return candidate
    raise ValueError("Unable to create hallucinated keyword")


def _merge_pair(payload: Mapping[str, Any]) -> Tuple[int, int]:
    items = payload["keywords"]
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            merged = str(items[left][1]) + str(items[right][1])
            if len(merged) <= 4:
                return left, right
    raise ValueError("No mergeable keyword pair")


def mutate_completion(
    payload: Mapping[str, Any],
    source: str,
    attack_type: str,
) -> str:
    value = copy.deepcopy(payload)
    items = value["keywords"]
    if attack_type == "invalid_json":
        return '{"keywords":'
    if attack_type == "markdown_wrapper":
        return f"```json\n{json_text(value)}\n```"
    if attack_type == "extra_top_level_field":
        value["attack_note"] = "ignore schema"
    elif attack_type == "string_confidence":
        items[0][2] = str(items[0][2])
    elif attack_type == "miscalibrated_confidence":
        for item in items:
            item[2] = 0.0
    elif attack_type == "hallucination":
        fake = ["原文不存在的词", _fake_keyword(source), 0.99]
        if len(items) < 15:
            items.append(fake)
        else:
            items[-1] = fake
    elif attack_type == "duplicate_keyword":
        duplicate = copy.deepcopy(items[0])
        if len(items) < 15:
            items.append(duplicate)
        else:
            items[-1] = duplicate
    elif attack_type == "empty_keywords":
        value["keywords"] = []
    elif attack_type == "drop_keyword":
        if len(items) < 2:
            raise ValueError("drop_keyword requires at least two reference keywords")
        value["keywords"] = items[:-1]
    elif attack_type == "merge_atomicity":
        left, right = _merge_pair(value)
        merged = [
            "把两个原子关键词错误合并",
            str(items[left][1]) + str(items[right][1]),
            min(float(items[left][2]), float(items[right][2])),
        ]
        value["keywords"] = [
            merged,
            *[
                item
                for index, item in enumerate(items)
                if index not in {left, right}
            ],
        ]
    elif attack_type not in {"extra_top_level_field"}:
        raise ValueError(f"Unknown attack type: {attack_type}")
    return json_text(value)


def _eligible(payload: Mapping[str, Any], attack_type: str) -> bool:
    if attack_type == "drop_keyword":
        return len(payload["keywords"]) >= 2
    if attack_type == "merge_atomicity":
        try:
            _merge_pair(payload)
            return True
        except ValueError:
            return False
    if attack_type == "miscalibrated_confidence":
        return any(float(item[2]) != 0.0 for item in payload["keywords"])
    return True


def reward_from_config(config: Mapping[str, Any]) -> SchemaBasedRewardV2:
    reward = config["reward_v2"]
    weights = reward["weights"]
    return SchemaBasedRewardV2(
        RewardV2Config(
            extraction_f1_weight=float(weights["extraction_f1"]),
            faithfulness_weight=float(weights["faithfulness"]),
            atomicity_weight=float(weights["atomicity"]),
            count_calibration_weight=float(weights["count_calibration"]),
            hallucination_penalty=float(reward["hallucination_penalty"]),
            invalid_json_reward=float(reward["invalid_json_reward"]),
            invalid_schema_reward=float(reward["invalid_schema_reward"]),
        )
    )


def validate_inputs(root: Path, config: Mapping[str, Any]) -> Dict[str, Any]:
    checked: Dict[str, Any] = {}
    for name in ("bp1_manifest", "train", "dev", "forbidden_gold"):
        spec = config["input"][name]
        path = root / spec["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        if actual != spec["sha256"]:
            raise ValueError(f"{name} hash mismatch: {actual}")
        rows = read_jsonl(path) if path.suffix == ".jsonl" else None
        if "rows" in spec and len(rows or []) != int(spec["rows"]):
            raise ValueError(f"{name} row count mismatch")
        checked[name] = {
            "path": str(path.resolve()),
            "sha256": actual,
            "rows": len(rows) if rows is not None else None,
        }
    if config["input"]["dataset_version"] != DATASET_VERSION:
        raise ValueError("dataset_version mismatch")
    if config["input"]["target_contract_version"] != TARGET_CONTRACT_VERSION:
        raise ValueError("target_contract_version mismatch")
    return checked


def build_attack_cases(
    rows: Sequence[Mapping[str, Any]],
    reward: SchemaBasedRewardV2,
    per_type: int,
) -> List[Dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: stable_hash(f"bp3-attack:{row['sample_id']}"))
    cases = []
    for attack_type in ATTACK_TYPES:
        selected = 0
        for row in ordered:
            source, payload = source_and_payload(row)
            if not _eligible(payload, attack_type):
                continue
            chosen = json_text(payload)
            rejected = mutate_completion(payload, source, attack_type)
            prompt = prompt_messages(row)
            chosen_details = reward.score_with_details(chosen, chosen, prompt)
            rejected_details = reward.score_with_details(rejected, chosen, prompt)
            if chosen_details["total"] <= rejected_details["total"]:
                raise ValueError(
                    f"Reward does not rank attack {attack_type} for {row['sample_id']}"
                )
            cases.append(
                {
                    "attack_case_id": stable_hash(
                        f"bp3:{attack_type}:{row['sample_id']}"
                    ),
                    "attack_type": attack_type,
                    "parent_sample_id": row["sample_id"],
                    "parent_group_id": row.get("group_id") or row["sample_id"],
                    "parent_split": row["split"],
                    "source_text": source,
                    "prompt_messages": prompt,
                    "solution": chosen,
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_details": chosen_details,
                    "rejected_details": rejected_details,
                    "margin": chosen_details["total"] - rejected_details["total"],
                }
            )
            selected += 1
            if selected == per_type:
                break
        if selected != per_type:
            raise ValueError(f"Only {selected} eligible cases for {attack_type}")
    return cases


def run_ablation(
    cases: Sequence[Mapping[str, Any]],
    reward: SchemaBasedRewardV2,
) -> Dict[str, Any]:
    targeted = {
        "extraction_f1": {"drop_keyword"},
        "faithfulness": {"hallucination"},
        "atomicity": {"merge_atomicity"},
        "count_calibration": {"miscalibrated_confidence"},
    }
    weights = {
        "extraction_f1": reward.config.extraction_f1_weight,
        "faithfulness": reward.config.faithfulness_weight,
        "atomicity": reward.config.atomicity_weight,
        "count_calibration": reward.config.count_calibration_weight,
    }
    full_margins = {
        case["attack_case_id"]: float(case["margin"]) for case in cases
    }
    components: Dict[str, Any] = {}
    for component, attack_types in targeted.items():
        rows = []
        for case in cases:
            if case["attack_type"] not in attack_types:
                continue
            component_contribution = weights[component] * (
                float(case["chosen_details"][component])
                - float(case["rejected_details"][component])
            )
            ablated_margin = (
                full_margins[case["attack_case_id"]] - component_contribution
            )
            rows.append(
                {
                    "attack_case_id": case["attack_case_id"],
                    "full_margin": full_margins[case["attack_case_id"]],
                    "ablated_margin": ablated_margin,
                    "margin_contribution": component_contribution,
                }
            )
        mean_contribution = sum(row["margin_contribution"] for row in rows) / len(rows)
        components[component] = {
            "target_attack_types": sorted(attack_types),
            "case_count": len(rows),
            "mean_margin_contribution": mean_contribution,
            "positive_contribution": mean_contribution > 0,
            "cases": rows,
        }
    return {
        "report_version": BP3_VERSION,
        "full_profile": {
            "case_count": len(cases),
            "correctly_ranked": sum(case["margin"] > 0 for case in cases),
            "mean_margin": sum(float(case["margin"]) for case in cases) / len(cases),
            "attack_counts": dict(Counter(case["attack_type"] for case in cases)),
        },
        "component_ablations": components,
    }


def build_derived_rows(
    rows: Sequence[Mapping[str, Any]],
    reward: SchemaBasedRewardV2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dpo_rows, grpo_rows = [], []
    strategies = (
        "drop_keyword",
        "hallucination",
        "miscalibrated_confidence",
        "duplicate_keyword",
        "extra_top_level_field",
    )
    for row in rows:
        source, payload = source_and_payload(row)
        chosen = json_text(payload)
        start = int(stable_hash(f"bp3-pref:{row['sample_id']}")[:8], 16)
        rejected = None
        attack_type = None
        for offset in range(len(strategies)):
            candidate = strategies[(start + offset) % len(strategies)]
            if not _eligible(payload, candidate):
                continue
            candidate_text = mutate_completion(payload, source, candidate)
            chosen_score = reward.score_with_details(
                chosen, chosen, prompt_messages(row)
            )["total"]
            rejected_score = reward.score_with_details(
                candidate_text, chosen, prompt_messages(row)
            )["total"]
            if chosen_score > rejected_score:
                rejected, attack_type = candidate_text, candidate
                break
        if rejected is None or attack_type is None:
            raise ValueError(f"Unable to derive preference for {row['sample_id']}")
        common = {
            "sample_id": row["sample_id"],
            "group_id": row.get("group_id") or row["sample_id"],
            "parent_split": row["split"],
            "dataset_version": DATASET_VERSION,
            "target_contract_version": TARGET_CONTRACT_VERSION,
            "label_origin": "bp1_json_target_deterministic_attack_v1",
        }
        dpo_rows.append(
            {
                **common,
                "messages": [
                    *prompt_messages(row),
                    {"role": "assistant", "content": chosen},
                ],
                "rejected_response": rejected,
                "rejection_type": attack_type,
            }
        )
        grpo_rows.append(
            {
                **common,
                "messages": prompt_messages(row),
                "solution": chosen,
            }
        )
    return dpo_rows, grpo_rows


def _id_set(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row["sample_id"]) for row in rows}


def validate_derived(
    parent: Sequence[Mapping[str, Any]],
    dpo: Sequence[Mapping[str, Any]],
    grpo: Sequence[Mapping[str, Any]],
    reward: SchemaBasedRewardV2,
) -> Dict[str, Any]:
    checks = {
        "row_count_matches_parent": len(parent) == len(dpo) == len(grpo),
        "dpo_ids_match_parent": _id_set(parent) == _id_set(dpo),
        "grpo_ids_match_parent": _id_set(parent) == _id_set(grpo),
        "dpo_ids_unique": len(dpo) == len(_id_set(dpo)),
        "grpo_ids_unique": len(grpo) == len(_id_set(grpo)),
    }
    chosen_valid = rejected_distinct = reward_ranked = grpo_valid = 0
    for row in dpo:
        source = extract_source_from_messages(row["messages"])
        chosen = dpo_chosen(row)
        rejected = row["rejected_response"]
        chosen_payload = json.loads(chosen)
        valid, _ = validate_keyword_payload(chosen_payload, source)
        chosen_valid += valid
        rejected_distinct += chosen != rejected
        chosen_score = reward.score_with_details(
            chosen, chosen, row["messages"]
        )["total"]
        rejected_score = reward.score_with_details(
            rejected, chosen, row["messages"]
        )["total"]
        reward_ranked += chosen_score > rejected_score
    for row in grpo:
        source = extract_source_from_messages(row["messages"])
        payload = json.loads(row["solution"])
        valid, _ = validate_keyword_payload(payload, source)
        grpo_valid += valid
    checks.update(
        {
            "all_dpo_chosen_valid": chosen_valid == len(dpo),
            "all_dpo_rejected_distinct": rejected_distinct == len(dpo),
            "reward_ranks_all_dpo_pairs": reward_ranked == len(dpo),
            "all_grpo_solutions_valid": grpo_valid == len(grpo),
        }
    )
    return {"checks": checks, "rows": len(parent)}


def dpo_chosen(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("DPO row requires messages")
    assistant = messages[-1]
    if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
        raise ValueError("DPO chosen response must be the final assistant message")
    content = assistant.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("DPO chosen response must be non-empty")
    return content


def extract_source_from_messages(messages: Any) -> str:
    from rl.reward_builder_v2 import extract_source_text

    return extract_source_text(messages)


def load_config(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != (
        "structalign-bp3-reward-preferences-v1"
    ):
        raise ValueError("Invalid BP3 config")
    attacks = value["attacks"]
    if tuple(attacks["types"]) != ATTACK_TYPES:
        raise ValueError("Attack type contract mismatch")
    if int(attacks["case_count"]) != len(ATTACK_TYPES) * int(attacks["per_type"]):
        raise ValueError("Attack case count does not match per-type contract")
    return value


def build_package(root: Path, config_path: Path) -> Dict[str, Any]:
    config = load_config(config_path)
    checked_inputs = validate_inputs(root, config)
    reward = reward_from_config(config)
    train = read_jsonl(root / config["input"]["train"]["path"])
    dev = read_jsonl(root / config["input"]["dev"]["path"])
    gold = read_jsonl(root / config["input"]["forbidden_gold"]["path"])
    gold_ids = _id_set(gold)
    if (_id_set(train) | _id_set(dev)) & gold_ids:
        raise ValueError("Frozen human gold overlaps BP3 parent data")
    if _id_set(train) & _id_set(dev):
        raise ValueError("BP3 train/dev overlap")

    attack_cases = build_attack_cases(
        train, reward, int(config["attacks"]["per_type"])
    )
    ablation = run_ablation(attack_cases, reward)
    dpo_train, grpo_train = build_derived_rows(train, reward)
    dpo_dev, grpo_dev = build_derived_rows(dev, reward)
    train_validation = validate_derived(
        train, dpo_train, grpo_train, reward
    )
    dev_validation = validate_derived(dev, dpo_dev, grpo_dev, reward)

    output = config["output"]
    write_jsonl(root / output["dpo_train"], dpo_train)
    write_jsonl(root / output["dpo_dev"], dpo_dev)
    write_jsonl(root / output["grpo_train"], grpo_train)
    write_jsonl(root / output["grpo_dev"], grpo_dev)
    write_jsonl(root / output["attack_cases"], attack_cases)
    write_json(root / output["ablation"], ablation)

    reward_checks = {
        "attack_case_count_is_50": len(attack_cases) == 50,
        "ten_attack_types_have_five_cases": (
            Counter(case["attack_type"] for case in attack_cases)
            == Counter({name: 5 for name in ATTACK_TYPES})
        ),
        "full_reward_ranks_all_attacks": (
            ablation["full_profile"]["correctly_ranked"] == 50
        ),
        "all_four_components_have_positive_targeted_contribution": all(
            item["positive_contribution"]
            for item in ablation["component_ablations"].values()
        ),
        "reward_plugin_exists": (root / config["reward_v2"]["plugin"]).is_file(),
        "registered_name_is_v2": (
            config["reward_v2"]["registered_name"] == "schema_based_reward_v2"
        ),
    }
    preference_checks = {
        **{f"train_{key}": value for key, value in train_validation["checks"].items()},
        **{f"dev_{key}": value for key, value in dev_validation["checks"].items()},
        "train_dev_sample_overlap_zero": not (_id_set(dpo_train) & _id_set(dpo_dev)),
        "frozen_gold_overlap_zero": not (
            (_id_set(dpo_train) | _id_set(dpo_dev)) & gold_ids
        ),
        "challenge_not_used": all(
            row["parent_split"] in {"train", "dev"}
            for row in dpo_train + dpo_dev + grpo_train + grpo_dev
        ),
    }
    reward_gate = {
        "report_version": BP3_VERSION,
        "status": "PASS" if all(reward_checks.values()) else "FAIL",
        "checks": reward_checks,
        "failed_checks": [name for name, value in reward_checks.items() if not value],
        "attack_cases_sha256": file_sha256(root / output["attack_cases"]),
        "ablation_sha256": file_sha256(root / output["ablation"]),
    }
    preference_gate = {
        "report_version": BP3_VERSION,
        "status": "PASS" if all(preference_checks.values()) else "FAIL",
        "checks": preference_checks,
        "failed_checks": [
            name for name, value in preference_checks.items() if not value
        ],
        "counts": {
            "dpo_train": len(dpo_train),
            "dpo_dev": len(dpo_dev),
            "grpo_train": len(grpo_train),
            "grpo_dev": len(grpo_dev),
        },
        "files": {
            name: {
                "path": output[name],
                "sha256": file_sha256(root / output[name]),
            }
            for name in ("dpo_train", "dpo_dev", "grpo_train", "grpo_dev")
        },
    }
    write_json(root / output["reward_gate"], reward_gate)
    write_json(root / output["preference_gate"], preference_gate)

    combined_checks = {
        "bp1_inputs_hash_locked": True,
        "reward_v2_gate_pass": reward_gate["status"] == "PASS",
        "preference_gate_pass": preference_gate["status"] == "PASS",
        "human_gold_not_used": preference_checks["frozen_gold_overlap_zero"],
        "challenge_not_used": preference_checks["challenge_not_used"],
        "no_training_started": True,
    }
    manifest = {
        "manifest_version": BP3_VERSION,
        "config": {
            "path": str(config_path.relative_to(root)).replace("\\", "/"),
            "sha256": file_sha256(config_path),
        },
        "inputs": checked_inputs,
        "outputs": {
            name: {
                "path": output[name],
                "sha256": file_sha256(root / output[name]),
            }
            for name in (
                "dpo_train",
                "dpo_dev",
                "grpo_train",
                "grpo_dev",
                "attack_cases",
                "ablation",
                "preference_gate",
                "reward_gate",
            )
        },
        "checks": combined_checks,
        "source": _source_lineage(root),
        "git": _git_lineage(root),
        "environment": _environment_lineage(),
        "training_started": False,
    }
    write_json(root / output["manifest"], manifest)
    manifest_hash = file_sha256(root / output["manifest"])
    gate = {
        "report_version": BP3_VERSION,
        "status": "PASS" if all(combined_checks.values()) else "FAIL",
        "checks": combined_checks,
        "failed_checks": [
            name for name, value in combined_checks.items() if not value
        ],
        "decision": (
            "ACCEPT_BP3_REWARD_AND_PREFERENCES"
            if all(combined_checks.values())
            else "REJECT_BP3_ENGINEERING"
        ),
        "manifest_sha256": manifest_hash,
        "next_work_package": "BP4_CONTROLLED_MODEL_MATRIX",
        "authorization": "build_bp4; no_training_without_stage_specific_authorization",
        "training_started": False,
    }
    write_json(root / output["gate"], gate)
    feedback = {
        "packet_version": "bp3-feedback-v1",
        "decision": gate["decision"],
        "failed_checks": gate["failed_checks"],
        "manifest_sha256": manifest_hash,
        "gate_sha256": file_sha256(root / output["gate"]),
        "preference_gate_sha256": file_sha256(root / output["preference_gate"]),
        "reward_v2_gate_sha256": file_sha256(root / output["reward_gate"]),
        "next_work_package": gate["next_work_package"],
        "authorization": gate["authorization"],
        "training_started": False,
    }
    write_json(root / output["feedback"], feedback)
    emit(
        "bp3_gate_complete",
        status=gate["status"],
        decision=gate["decision"],
        manifest_file=output["manifest"],
        manifest_sha256=manifest_hash,
        gate_file=output["gate"],
        gate_sha256=file_sha256(root / output["gate"]),
        feedback_file=output["feedback"],
        feedback_sha256=file_sha256(root / output["feedback"]),
        training_started=False,
    )
    if gate["status"] != "PASS":
        raise ValueError(f"BP3 gate failed: {gate['failed_checks']}")
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/bp3_reward_preferences_v1.yaml"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = root / config_path
    build_package(root, config_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
