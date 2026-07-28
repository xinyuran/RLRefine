#!/usr/bin/env python3
import json

from core.config import Config
from core.keyword_contract import MAX_KEYWORD_LENGTH, MAX_KEYWORDS
from examples.keyword_extraction.schema import create_keyword_schema
from rl.reward_builder import RewardBuilder


def emit(event, **payload):
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True))


def main() -> int:
    schema = create_keyword_schema()
    schema_json = schema.to_json_schema()
    reward = RewardBuilder.create_keyword_reward()
    config = Config()
    valid_example = {"keywords": [["属性", "屏幕", 0.9]]}
    invalid_examples = [
        {"keywords": [["属性", "配送速度慢", 0.9]]},
        {"keywords": [["属性", "屏幕", "0.9"]]},
        {"keywords": [{"category": "属性", "keyword": "屏幕", "confidence": 0.9}]},
    ]
    valid_ok, valid_errors = schema.validate(valid_example)
    invalid_results = [schema.validate(item)[0] for item in invalid_examples]
    checks = {
        "valid_tuple_accepted": valid_ok,
        "invalid_examples_rejected": not any(invalid_results),
        "post_process_max_keyword_length": config.post_process_max_keyword_length == MAX_KEYWORD_LENGTH,
        "reward_max_keyword_length": reward.config.max_keyword_length == MAX_KEYWORD_LENGTH,
        "reward_max_items": reward.config.max_items == MAX_KEYWORDS,
        "reward_schema_max_items": reward.schema["properties"]["keywords"].get("maxItems") == MAX_KEYWORDS,
        "reward_schema_keyword_length": (
            reward.schema["properties"]["keywords"]["items"]["prefixItems"][1].get("maxLength")
            == MAX_KEYWORD_LENGTH
        ),
        "schema_max_items": schema_json["properties"]["keywords"].get("maxItems") == MAX_KEYWORDS,
        "schema_confidence_type_number": (
            schema_json["properties"]["keywords"]["items"]["prefixItems"][2].get("type") == "number"
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    emit(
        "p0_schema_contract_complete",
        status=status,
        checks=checks,
        max_keyword_length=MAX_KEYWORD_LENGTH,
        max_keywords=MAX_KEYWORDS,
        tuple_size=3,
        valid_example_errors=valid_errors,
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
