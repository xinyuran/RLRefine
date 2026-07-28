"""Frozen E4 vLLM benchmark: unconstrained versus strict JSON Schema serving."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from evaluation.bp2_evaluator import prepare_reference
from evaluation.bp4_base_v2 import parse_completion, prepare_inference_rows
from evaluation.keyword_evaluator import evaluate_records, file_sha256, read_jsonl, write_json_atomic, write_jsonl_atomic
from evaluation.structured_decoding_inference import keyword_response_format


CONCURRENCIES = (1, 4, 8)
WARMUP_ROWS = 10


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    values = sorted(values); index = min(len(values) - 1, round((len(values) - 1) * percentile))
    return values[index]


def _request(client: Any, row: dict[str, Any], model: str, constrained: bool) -> dict[str, Any]:
    started = time.perf_counter(); first_token = None; chunks = []
    options: dict[str, Any] = {"model": model, "messages": row["messages"], "temperature": 0, "seed": 42, "max_tokens": 768, "stream": True, "stream_options": {"include_usage": True}}
    if constrained: options["response_format"] = keyword_response_format()
    stream = client.chat.completions.create(**options)
    finish_reason = None; usage = None
    for chunk in stream:
        if first_token is None: first_token = time.perf_counter()
        choice = chunk.choices[0] if chunk.choices else None
        if choice:
            if choice.delta.content: chunks.append(choice.delta.content)
            finish_reason = choice.finish_reason or finish_reason
        usage = getattr(chunk, "usage", None) or usage
    elapsed = time.perf_counter() - started
    raw_response = "".join(chunks)
    prediction = parse_completion(row["sample_id"], raw_response, row["source"])
    prediction.update({"input_tokens": getattr(usage, "prompt_tokens", None), "output_tokens": getattr(usage, "completion_tokens", None), "finish_reason": finish_reason or "unknown", "ttft_seconds": (first_token or time.perf_counter()) - started, "request_seconds": elapsed, "raw_response": raw_response})
    if not isinstance(prediction["input_tokens"], int) or not isinstance(prediction["output_tokens"], int):
        raise ValueError("vLLM streaming response omitted token usage; start server with --enable-prompt-tokens-details if needed")
    return prediction


def _run_arm(client: Any, rows: list[dict[str, Any]], model: str, constrained: bool, concurrency: int) -> tuple[list[dict[str, Any]], float]:
    # Warm-up is deliberately excluded from measurements and never written as predictions.
    for row in rows[:WARMUP_ROWS]: _request(client, row, model, constrained)
    arm_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_request, client, row, model, constrained): row["sample_id"] for row in rows}
        predictions = [future.result() for future in as_completed(futures)]
    wall_clock_seconds = time.perf_counter() - arm_started
    return sorted(predictions, key=lambda item: item["sample_id"]), wall_clock_seconds


def _metrics(reference: list[dict[str, Any]], predictions: list[dict[str, Any]], wall_clock_seconds: float) -> dict[str, Any]:
    report, _ = evaluate_records(reference, predictions, "E4_GRPO_FROM_DPO")
    request_seconds = [row["request_seconds"] for row in predictions]; ttft = [row["ttft_seconds"] for row in predictions]
    total_tokens = sum(row["output_tokens"] for row in predictions)
    return {"macro_f1": report["macro"]["f1"], "micro_f1": report["micro"]["f1"], "schema_valid_rate": report["schema_valid_rate"], "hallucination_keyword_rate": report["hallucination"]["keyword_rate"], "target_contract_error_count": sum(row["status"] == "error" for row in predictions), "transport_error_count": 0, "max_token_output_count": sum(row["output_tokens"] >= 768 for row in predictions), "ttft_p50_seconds": _percentile(ttft, .5), "ttft_p95_seconds": _percentile(ttft, .95), "e2e_p50_seconds": _percentile(request_seconds, .5), "e2e_p95_seconds": _percentile(request_seconds, .95), "wall_clock_seconds": wall_clock_seconds, "throughput_output_tokens_per_wall_second": total_tokens / max(wall_clock_seconds, 1e-12), "mean_tpot_seconds": sum(max(row["request_seconds"] - row["ttft_seconds"], 0) / max(row["output_tokens"] - 1, 1) for row in predictions) / len(predictions)}


def run(args: argparse.Namespace) -> int:
    from openai import OpenAI
    freeze = json.loads(args.e4_freeze.read_text(encoding="utf-8"))
    if freeze.get("decision") != "FREEZE_E4_AS_OPTIMAL_RESEARCH_CANDIDATE": raise ValueError("Serving requires a frozen E4 research candidate")
    rows = prepare_inference_rows(read_jsonl(args.dev_file), "dev")
    reference = prepare_reference(read_jsonl(args.dev_file))
    if len(rows) != 331 or len(reference) != 331: raise ValueError("Serving matrix requires the frozen 331-row teacher-dev split")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=args.base_url.rstrip("/") + "/v1", api_key=args.api_key, timeout=args.timeout_seconds, max_retries=0)
    available = [item.id for item in client.models.list().data]
    if args.model not in available: raise ValueError(f"Served E4 model missing: {args.model!r}; available={available}")
    manifest = {"report_version": "bp5-e4-serving-matrix-v2", "e4_freeze_sha256": file_sha256(args.e4_freeze), "dev_sha256": file_sha256(args.dev_file), "sample_count": 331, "model": args.model, "base_url": args.base_url, "decoding": {"temperature": 0, "seed": 42, "max_tokens": 768}, "warmup_rows": WARMUP_ROWS, "concurrencies": list(CONCURRENCIES), "human_gold_used": False, "single_variable": "response_format_json_schema"}
    write_json_atomic(args.output_dir / "run_manifest.json", manifest)
    results = {}
    for constrained in (False, True):
        arm = "vllm_json_schema" if constrained else "vllm_unconstrained"
        for concurrency in CONCURRENCIES:
            predictions, wall_clock_seconds = _run_arm(client, rows, args.model, constrained, concurrency)
            path = args.output_dir / f"{arm}_c{concurrency}_predictions.jsonl"; write_jsonl_atomic(path, predictions)
            results[f"{arm}_c{concurrency}"] = {**_metrics(reference, predictions, wall_clock_seconds), "predictions_sha256": file_sha256(path)}
    unconstrained = results["vllm_unconstrained_c4"]; constrained = results["vllm_json_schema_c4"]
    checks = {"schema_valid_rate_at_least_99_percent": constrained["schema_valid_rate"] >= .99, "macro_f1_within_0_03_of_unconstrained": constrained["macro_f1"] >= unconstrained["macro_f1"] - .03, "no_transport_errors": constrained["transport_error_count"] == 0, "no_target_contract_errors": constrained["target_contract_error_count"] == 0, "no_max_token_outputs": constrained["max_token_output_count"] == 0, "concurrency_4_schema_matches_concurrency_1": results["vllm_json_schema_c4"]["schema_valid_rate"] >= results["vllm_json_schema_c1"]["schema_valid_rate"], "concurrency_4_macro_within_0_03_of_concurrency_1": results["vllm_json_schema_c4"]["macro_f1"] >= results["vllm_json_schema_c1"]["macro_f1"] - .03}
    report = {"status": "PASS", "report_version": "bp5-e4-serving-benchmark-v2", "human_gold_used": False, "results": results, "checks": checks, "decision": "ACCEPT_E4_CONSTRAINED_SERVING_CANDIDATE" if all(checks.values()) else "RETAIN_E4_OFFLINE_BATCH_CANDIDATE", "deployment_authorized": False}
    write_json_atomic(args.output_dir / "serving_benchmark.json", report)
    print(json.dumps({"event": "bp5_e4_serving_benchmark_complete", **report}, ensure_ascii=False, sort_keys=True)); return 0


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--base-url", default="http://127.0.0.1:8002"); parser.add_argument("--api-key", default="EMPTY"); parser.add_argument("--model", required=True); parser.add_argument("--e4-freeze", type=Path, default=Path("reports/model_gates/bp5_e4_research_candidate_freeze.json")); parser.add_argument("--dev-file", type=Path, default=Path("data/canonical/keyword_v2/dev.jsonl")); parser.add_argument("--output-dir", type=Path, default=Path("reports/bp5/serving")); parser.add_argument("--timeout-seconds", type=float, default=300)
    args = parser.parse_args()
    try: return run(args)
    except Exception as exc:
        print(json.dumps({"event": "bp5_e4_serving_benchmark_complete", "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, sort_keys=True)); return 1


if __name__ == "__main__": raise SystemExit(main())
