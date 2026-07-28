"""Versioned training/inference target contract for keyword extraction."""
import json
from typing import Any, Dict, List, Sequence, Tuple

from core.keyword_contract import KEYWORD_TUPLE_SIZE, MAX_KEYWORD_LENGTH, MAX_KEYWORDS


TARGET_CONTRACT_VERSION = "keyword-json-target-v2"
DATASET_VERSION = "keyword-v2.0.0"

SYSTEM_PROMPT = f"""你是中文电商评论关键词抽取器。训练和推理必须遵循同一协议。

【抽取规则】
1. 关键词必须是原文中的连续子串，禁止改写、归纳或补充原文没有的词。
2. 对象词与描述词分别输出；否定描述保留紧凑否定结构。
3. 纯日期、时长、编号通常不作为关键词。
4. 每个关键词为1至{MAX_KEYWORD_LENGTH}个汉字，去重后最多{MAX_KEYWORDS}项，按重要性降序排列。

【输出规则】
1. 在内部完成分析，但不要输出思考过程、Markdown或代码围栏。
2. 只输出一个JSON对象，顶层只能有`keywords`。
3. `keywords`每项必须恰好是{KEYWORD_TUPLE_SIZE}元组：
   ["原文依据的简短说明", "关键词", 0到1之间的数字]
4. 说明只解释当前单个关键词；置信度必须是JSON number，不能是字符串。
5. 输出前检查JSON、数量、长度、去重、三元组形状和原文连续子串约束。"""


def build_messages(source_text: str, assistant_payload: Dict[str, Any] | None = None) -> List[Dict[str, str]]:
    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError("source_text must be a non-empty string")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "请按统一协议抽取下面评论的关键词。\n\n"
                f"【待处理评论】\n{source_text.strip()}\n\n"
                "请严格只输出合法JSON对象。"
            ),
        },
    ]
    if assistant_payload is not None:
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    assistant_payload, ensure_ascii=False, separators=(",", ":")
                ),
            }
        )
    return messages


def validate_keyword_payload(payload: Any, source_text: str) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(payload, dict) or set(payload) != {"keywords"}:
        return False, ["payload_top_level_invalid"]
    keywords = payload.get("keywords")
    if not isinstance(keywords, list) or not 1 <= len(keywords) <= MAX_KEYWORDS:
        return False, ["keyword_count_invalid"]
    seen = set()
    for index, item in enumerate(keywords):
        prefix = f"keywords[{index}]"
        if not isinstance(item, list) or len(item) != KEYWORD_TUPLE_SIZE:
            errors.append(f"{prefix}.shape_invalid")
            continue
        explanation, keyword, confidence = item
        if not isinstance(explanation, str) or not explanation.strip():
            errors.append(f"{prefix}.explanation_invalid")
        if not isinstance(keyword, str) or not 1 <= len(keyword) <= MAX_KEYWORD_LENGTH:
            errors.append(f"{prefix}.keyword_invalid")
        else:
            if keyword not in source_text:
                errors.append(f"{prefix}.keyword_not_in_source")
            if keyword in seen:
                errors.append(f"{prefix}.keyword_duplicate")
            seen.add(keyword)
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= float(confidence) <= 1
        ):
            errors.append(f"{prefix}.confidence_invalid")
    return not errors, errors


def keyword_values(payload: Dict[str, Any]) -> Sequence[str]:
    return [item[1] for item in payload["keywords"]]
