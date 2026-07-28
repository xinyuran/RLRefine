#!/usr/bin/env python3
import json

from core.config import Config
from core.processor import RLRefineProcessor
from examples.keyword_extraction.schema import create_keyword_task


def emit(event, **payload):
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True))


def main() -> int:
    task = create_keyword_task()
    processor = RLRefineProcessor.__new__(RLRefineProcessor)
    processor.config = Config(task_schema=task.schema, task=task)
    processor.config.enable_preprocess = False
    processor.task = task
    processor.prompt_builder = None
    processor._call_llm = lambda *args, **kwargs: json.dumps(
        {"keywords": [["属性", "屏幕", 0.9]]}, ensure_ascii=False
    )
    processor._fallback_extract = lambda text: []
    result = processor.process_single("屏幕清晰", "contract-sample")
    checks = {
        "status_success": result.get("status") == "success",
        "fallback_false": result.get("fallback") is False,
        "tuple_preserved": result.get("data", {}).get("keywords") == [["属性", "屏幕", 0.9]],
        "full_info_default": processor.config.post_process_return_full_info is True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    emit("p0_processor_contract_complete", status=status, checks=checks)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
