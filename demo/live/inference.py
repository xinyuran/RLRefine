"""Input validation and presentation for the live model demo."""
import json
import math
import re

from core.target_contract import validate_keyword_payload
from prompts.prompt_template_3 import get_keyword_extraction_prompt_3


MAX_SOURCE_LENGTH = 512


def prepare_messages(source):
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Enter a review first / 请先输入原文。")
    if len(source) > MAX_SOURCE_LENGTH:
        raise ValueError(f"Use at most {MAX_SOURCE_LENGTH} characters / 原文最多 {MAX_SOURCE_LENGTH} 字符。")
    system, user = get_keyword_extraction_prompt_3(source)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_model_json(raw):
    """Read the final keyword object after the model's legacy analysis prefix."""
    def finite_number(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Non-finite JSON number: {value}")
        return number

    decoder = json.JSONDecoder(parse_constant=finite_number, parse_float=finite_number)
    candidate = None
    for match in re.finditer(r"\{", raw):
        try:
            value, end = decoder.raw_decode(raw, match.start())
        except ValueError:
            continue
        if isinstance(value, dict) and "keywords" in value and raw[end:].strip() in ("", "```"):
            candidate = value
    return candidate


def inspect_response(source, raw, completed=True):
    payload = parse_model_json(raw)
    if payload is None:
        errors = ["No complete keyword JSON object was generated."]
    else:
        _, errors = validate_keyword_payload(payload, source)
    raw_errors = list(errors)
    normalizations = []
    rows, words = [], []
    items = payload.get("keywords") if isinstance(payload, dict) else None
    for index, item in enumerate(items if isinstance(items, list) else []):
        if not isinstance(item, list) or len(item) != 3:
            continue
        explanation, keyword, confidence = item
        if isinstance(confidence, str):
            try:
                numeric_score = float(confidence)
            except ValueError:
                pass
            else:
                if math.isfinite(numeric_score):
                    item[2] = confidence = numeric_score
                    normalizations.append(f"keywords[{index}].confidence_string_to_number")
        if not isinstance(keyword, str) or not keyword:
            continue
        rows.append([keyword, str(confidence), str(explanation), "Yes" if keyword in source else "No"])
        if keyword in source:
            words.append(keyword)

    if normalizations:
        _, errors = validate_keyword_payload(payload, source)
    if not completed:
        errors = ["Generation stopped before the model finished."] + errors

    covered = [False] * len(source)
    for word in set(words):
        start = source.find(word)
        while start != -1:
            covered[start:start + len(word)] = [True] * len(word)
            start = source.find(word, start + 1)
    highlights = []
    for character, matched in zip(source, covered):
        label = "Keyword" if matched else None
        if highlights and highlights[-1][1] == label:
            highlights[-1] = (highlights[-1][0] + character, label)
        else:
            highlights.append((character, label))

    return {
        "payload": payload, "errors": errors, "raw_errors": raw_errors,
        "normalizations": normalizations, "rows": rows, "highlights": highlights,
    }
