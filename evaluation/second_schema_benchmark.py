"""One-shot Base-model benchmark for support-ticket routing."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from evaluation.keyword_evaluator import file_sha256, read_jsonl, write_json_atomic, write_jsonl_atomic
from examples.intent_routing.schema import INTENTS, create_intent_routing_schema


CONCURRENCY = 4
MAX_TOKENS = 128
WARMUP_ROWS = 5
SPLITS = ("dev", "test", "challenge")

SYSTEM_PROMPT = """你是客服工单路由器。只输出一个 JSON 对象，不要解释。
intent 只能是 refund、delivery、product_quality、account、other 之一。
urgency 只能是 low、normal、high 之一。
evidence 必须逐字复制自用户工单，长度为 1 到 40 个字符。"""


def response_format() -> dict[str, Any]:
    schema = create_intent_routing_schema().to_json_schema()
    schema["additionalProperties"] = False
    return {
        "type": "json_schema",
        "json_schema": {"name": "support_ticket_routing", "strict": True, "schema": schema},
    }


def _messages(source: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"工单：{source}"},
    ]


def parse_prediction(sample_id: str, source: str, raw_response: str) -> dict[str, Any]:
    try:
        data = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as exc:
        return {
            "sample_id": sample_id,
            "status": "error",
            "data": None,
            "schema_valid": False,
            "target_contract_valid": False,
            "schema_errors": [f"invalid_json:{type(exc).__name__}"],
            "raw_response": raw_response,
        }
    schema = create_intent_routing_schema()
    valid, errors = schema.validate(data if isinstance(data, dict) else {})
    if isinstance(data, dict) and set(data) != {"intent", "urgency", "evidence"}:
        valid = False
        errors = [*errors, "unexpected_or_missing_keys"]
    faithful = valid and data["evidence"] in source
    return {
        "sample_id": sample_id,
        "status": "success" if faithful else "error",
        "data": data if isinstance(data, dict) else None,
        "schema_valid": valid,
        "target_contract_valid": faithful,
        "schema_errors": errors,
        "raw_response": raw_response,
    }


def _request(client: Any, row: dict[str, Any], model: str, constrained: bool) -> dict[str, Any]:
    started = time.perf_counter()
    first_token = None
    chunks: list[str] = []
    options: dict[str, Any] = {
        "model": model,
        "messages": _messages(row["source"]),
        "temperature": 0,
        "seed": 42,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if constrained:
        options["response_format"] = response_format()
    try:
        stream = client.chat.completions.create(**options)
        usage = None
        finish_reason = None
        for chunk in stream:
            if first_token is None:
                first_token = time.perf_counter()
            choice = chunk.choices[0] if chunk.choices else None
            if choice:
                if choice.delta.content:
                    chunks.append(choice.delta.content)
                finish_reason = choice.finish_reason or finish_reason
            usage = getattr(chunk, "usage", None) or usage
        elapsed = time.perf_counter() - started
        prediction = parse_prediction(row["sample_id"], row["source"], "".join(chunks))
        prediction.update({
            "transport_error": None,
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "finish_reason": finish_reason or "unknown",
            "ttft_seconds": (first_token or time.perf_counter()) - started,
            "request_seconds": elapsed,
        })
        return prediction
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {
            "sample_id": row["sample_id"],
            "status": "error",
            "data": None,
            "schema_valid": False,
            "target_contract_valid": False,
            "schema_errors": [],
            "raw_response": "",
            "transport_error": f"{type(exc).__name__}: {exc}",
            "input_tokens": None,
            "output_tokens": None,
            "finish_reason": "transport_error",
            "ttft_seconds": elapsed,
            "request_seconds": elapsed,
        }


def _run_rows(
    client: Any,
    rows: list[dict[str, Any]],
    model: str,
    constrained: bool,
    warmup: bool,
) -> tuple[list[dict[str, Any]], float]:
    if warmup:
        for row in rows[:WARMUP_ROWS]:
            result = _request(client, row, model, constrained)
            if result["transport_error"]:
                raise RuntimeError(f"warmup request failed: {result['transport_error']}")
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(_request, client, row, model, constrained) for row in rows]
        predictions = [future.result() for future in as_completed(futures)]
    return sorted(predictions, key=lambda row: row["sample_id"]), time.perf_counter() - started


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * percentile))]


def evaluate_records(
    gold_rows: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    wall_clock_seconds: float,
) -> dict[str, Any]:
    gold_by_id = {row["sample_id"]: row for row in gold_rows}
    pred_by_id = {row["sample_id"]: row for row in predictions}
    if len(gold_by_id) != len(gold_rows) or len(pred_by_id) != len(predictions):
        raise ValueError("duplicate sample_id in gold or predictions")
    if set(gold_by_id) != set(pred_by_id):
        raise ValueError("prediction sample_id coverage mismatch")

    confusion = {label: Counter() for label in INTENTS}
    urgency_correct = evidence_exact = faithful = schema_valid = contract_valid = 0
    transport_errors = target_errors = 0
    request_seconds: list[float] = []
    ttft_seconds: list[float] = []
    output_tokens = 0
    for sample_id, gold in gold_by_id.items():
        prediction = pred_by_id[sample_id]
        data = prediction.get("data") if isinstance(prediction.get("data"), dict) else {}
        gold_target = gold["target"]
        predicted_intent = data.get("intent")
        for label in INTENTS:
            if predicted_intent == label and gold_target["intent"] == label:
                confusion[label]["tp"] += 1
            elif predicted_intent == label:
                confusion[label]["fp"] += 1
            elif gold_target["intent"] == label:
                confusion[label]["fn"] += 1
        urgency_correct += data.get("urgency") == gold_target["urgency"]
        evidence_exact += data.get("evidence") == gold_target["evidence"]
        evidence = data.get("evidence")
        faithful += isinstance(evidence, str) and evidence in gold["source"]
        schema_valid += prediction.get("schema_valid") is True
        contract_valid += prediction.get("target_contract_valid") is True
        transport_errors += prediction.get("transport_error") is not None
        target_errors += prediction.get("transport_error") is None and prediction.get("target_contract_valid") is not True
        if isinstance(prediction.get("request_seconds"), (int, float)):
            request_seconds.append(float(prediction["request_seconds"]))
        if isinstance(prediction.get("ttft_seconds"), (int, float)):
            ttft_seconds.append(float(prediction["ttft_seconds"]))
        if isinstance(prediction.get("output_tokens"), int):
            output_tokens += prediction["output_tokens"]

    per_class_f1 = {}
    for label, counts in confusion.items():
        precision = _safe_ratio(counts["tp"], counts["tp"] + counts["fp"])
        recall = _safe_ratio(counts["tp"], counts["tp"] + counts["fn"])
        per_class_f1[label] = _safe_ratio(2 * precision * recall, precision + recall)
    sample_count = len(gold_rows)
    return {
        "sample_count": sample_count,
        "intent_macro_f1": statistics.fmean(per_class_f1.values()),
        "intent_per_class_f1": per_class_f1,
        "urgency_accuracy": _safe_ratio(urgency_correct, sample_count),
        "evidence_exact_rate": _safe_ratio(evidence_exact, sample_count),
        "evidence_source_faithfulness_rate": _safe_ratio(faithful, sample_count),
        "schema_valid_rate": _safe_ratio(schema_valid, sample_count),
        "target_contract_valid_rate": _safe_ratio(contract_valid, sample_count),
        "transport_error_count": transport_errors,
        "target_contract_error_count": target_errors,
        "e2e_p50_seconds": _percentile(request_seconds, 0.5),
        "e2e_p95_seconds": _percentile(request_seconds, 0.95),
        "ttft_p50_seconds": _percentile(ttft_seconds, 0.5),
        "ttft_p95_seconds": _percentile(ttft_seconds, 0.95),
        "wall_clock_seconds": wall_clock_seconds,
        "throughput_output_tokens_per_wall_second": _safe_ratio(output_tokens, wall_clock_seconds),
    }


def _validate_frozen_data(data_dir: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_flags = {
        "status": "FROZEN_FOR_ONE_SHOT_SMOKE_TEST",
        "synthetic": True,
        "human_verified": True,
        "frozen": True,
    }
    for key, expected in required_flags.items():
        if manifest.get(key) != expected:
            raise ValueError(f"frozen manifest requires {key}={expected!r}")
    rows_by_split = {}
    for split in SPLITS:
        path = data_dir / f"{split}.jsonl"
        metadata = manifest["files"][split]
        if file_sha256(path) != metadata["sha256"]:
            raise ValueError(f"{split} SHA256 differs from frozen manifest")
        rows = read_jsonl(path)
        if len(rows) != metadata["rows"]:
            raise ValueError(f"{split} row count differs from frozen manifest")
        if not all(row.get("human_verified") is True and row.get("annotation_status") == "frozen" for row in rows):
            raise ValueError(f"{split} contains non-frozen rows")
        rows_by_split[split] = rows
    return manifest, rows_by_split


def run(args: argparse.Namespace) -> int:
    from openai import OpenAI

    preregistration = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if preregistration.get("decision") != "RUN_EXACTLY_ONE_SECOND_SCHEMA_COMPARISON":
        raise ValueError("second-Schema preregistration is missing or inactive")
    if preregistration.get("arms") != ["base_unconstrained", "base_strict_json_schema"]:
        raise ValueError("second-Schema arms differ from preregistration")
    if preregistration.get("concurrency") != CONCURRENCY:
        raise ValueError("second-Schema concurrency differs from preregistration")
    model_contract = preregistration.get("model", {})
    if args.model != model_contract.get("model_id") and model_contract.get("revision") not in args.model:
        raise ValueError("served model id does not identify the preregistered Base revision")
    manifest, rows_by_split = _validate_frozen_data(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(
        base_url=args.base_url.rstrip("/") + "/v1",
        api_key=args.api_key,
        timeout=args.timeout_seconds,
        max_retries=0,
    )
    available = [item.id for item in client.models.list().data]
    if args.model not in available:
        raise ValueError(f"served Base model missing: {args.model!r}; available={available}")
    run_manifest = {
        "report_version": "second-schema-serving-v1",
        "model": args.model,
        "model_role": "Base_only",
        "base_url": args.base_url,
        "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        "preregistration_sha256": file_sha256(args.preregistration),
        "dataset_synthetic": manifest["synthetic"],
        "dataset_human_verified": manifest["human_verified"],
        "concurrency": CONCURRENCY,
        "warmup_rows": WARMUP_ROWS,
        "decoding": {"temperature": 0, "seed": 42, "max_tokens": MAX_TOKENS},
        "arms": ["unconstrained", "json_schema"],
        "single_variable": "response_format_json_schema",
        "keyword_e4_adapter_used": False,
        "training_used": False,
    }
    write_json_atomic(args.output_dir / "run_manifest.json", run_manifest)
    results: dict[str, Any] = {}
    for constrained in (False, True):
        arm = "json_schema" if constrained else "unconstrained"
        results[arm] = {}
        for split in SPLITS:
            predictions, wall_seconds = _run_rows(
                client, rows_by_split[split], args.model, constrained, warmup=split == "dev",
            )
            prediction_path = args.output_dir / f"{arm}_{split}_predictions.jsonl"
            write_jsonl_atomic(prediction_path, predictions)
            results[arm][split] = {
                **evaluate_records(rows_by_split[split], predictions, wall_seconds),
                "predictions_sha256": file_sha256(prediction_path),
            }
    baseline = results["unconstrained"]["test"]
    constrained = results["json_schema"]["test"]
    checks = {
        "test_schema_valid_rate_at_least_0_99": constrained["schema_valid_rate"] >= 0.99,
        "test_target_contract_valid_rate_at_least_0_95": constrained["target_contract_valid_rate"] >= 0.95,
        "test_intent_macro_f1_within_0_03_of_unconstrained": constrained["intent_macro_f1"] >= baseline["intent_macro_f1"] - 0.03,
        "test_no_transport_errors": constrained["transport_error_count"] == 0,
    }
    report = {
        "status": "PASS",
        "report_version": "second-schema-smoke-test-v1",
        "dataset_scope": "synthetic_human_verified_framework_adaptation_smoke_test",
        "results": results,
        "checks": checks,
        "decision": "SECOND_SCHEMA_SMOKE_TEST_PASS" if all(checks.values()) else "SECOND_SCHEMA_SMOKE_TEST_FAIL",
        "stop_rule_applied": True,
        "deployment_authorized": False,
    }
    write_json_atomic(args.output_dir / "benchmark.json", report)
    print(json.dumps({"event": "second_schema_benchmark_complete", **report}, ensure_ascii=False, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8003")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--data-dir", type=Path, default=Path("data/frozen/intent_routing_v1"))
    parser.add_argument("--preregistration", type=Path, default=Path("reports/second_schema/preregistration.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/second_schema/serving"))
    parser.add_argument("--timeout-seconds", type=float, default=300)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({"event": "second_schema_benchmark_complete", "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
