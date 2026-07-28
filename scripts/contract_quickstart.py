"""CPU-only smoke test for the public keyword and routing contracts."""
import json

from core.target_contract import validate_keyword_payload
from examples.intent_routing.schema import create_intent_routing_schema


def main() -> int:
    source = "包装很好，屏幕清晰。"
    keyword_payload = {
        "keywords": [
            ["原文评价包装", "包装", 0.96],
            ["原文描述屏幕", "屏幕", 0.94],
        ]
    }
    keyword_valid, keyword_errors = validate_keyword_payload(keyword_payload, source)

    routing_schema = create_intent_routing_schema()
    routing_payload = {
        "intent": "delivery",
        "urgency": "normal",
        "evidence": "快递一直没到",
    }
    routing_valid, routing_errors = routing_schema.validate(routing_payload)

    result = {
        "keyword_contract": {"valid": keyword_valid, "errors": keyword_errors},
        "routing_schema": {"valid": routing_valid, "errors": routing_errors},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if keyword_valid and routing_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
