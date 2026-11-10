"""Deterministic exact-set evaluation for keyword extraction predictions."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from examples.keyword_extraction.schema import create_keyword_schema


VALID_STATUSES = {"success", "fallback", "error"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _index_unique(rows: Sequence[Dict[str, Any]], source: str) -> Dict[str, Dict[str, Any]]:
    indexed = {}
    for row_number, row in enumerate(rows, 1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{source} row {row_number} has no non-empty sample_id")
        if sample_id in indexed:
            raise ValueError(f"Duplicate sample_id in {source}: {sample_id}")
        indexed[sample_id] = row
    return indexed


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _sample_scores(predicted: set, gold: set) -> Tuple[float, float, float]:
    true_positive = len(predicted & gold)
    precision = _safe_ratio(true_positive, len(predicted))
    recall = _safe_ratio(true_positive, len(gold))
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    return precision, recall, f1


def _prediction_keywords(row: Dict[str, Any]) -> Tuple[List[str], bool, List[str]]:
    status = row.get("status")
    if status not in VALID_STATUSES:
        return [], False, [f"Unsupported status: {status!r}"]
    if status == "error":
        if row.get("data") is not None:
            return [], False, ["error status requires data=null"]
        return [], False, []

    data = row.get("data")
    valid, errors = create_keyword_schema().validate(data if isinstance(data, dict) else {})
    if not valid:
        return [], False, errors
    return [item[1] for item in data["keywords"]], True, []


def evaluate_records(
    gold_rows: Sequence[Dict[str, Any]],
    prediction_rows: Sequence[Dict[str, Any]],
    model_id: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    gold_by_id = _index_unique(gold_rows, "gold")
    prediction_by_id = _index_unique(prediction_rows, "predictions")
    missing = sorted(set(gold_by_id) - set(prediction_by_id))
    extra = sorted(set(prediction_by_id) - set(gold_by_id))
    if missing or extra:
        raise ValueError(
            f"Prediction sample_id coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    status_counts = Counter()
    schema_valid_count = 0
    exact_count = 0
    total_tp = total_fp = total_fn = 0
    macro_precision = macro_recall = macro_f1 = 0.0
    predicted_unique_total = hallucinated_total = 0
    hallucinated_sample_count = 0
    duplicate_prediction_items = 0
    error_rows = []

    for sample_id in sorted(gold_by_id):
        gold_row = gold_by_id[sample_id]
        prediction_row = prediction_by_id[sample_id]
        gold_data = gold_row.get("gold")
        gold_keywords = gold_data.get("keywords") if isinstance(gold_data, dict) else None
        if not isinstance(gold_keywords, list) or not gold_keywords or not all(
            isinstance(item, str) and item for item in gold_keywords
        ):
            raise ValueError(f"Invalid gold keywords for sample_id: {sample_id}")
        if len(gold_keywords) != len(set(gold_keywords)):
            raise ValueError(f"Duplicate gold keywords for sample_id: {sample_id}")

        status = prediction_row.get("status")
        status_counts[str(status)] += 1
        predicted_items, schema_valid, schema_errors = _prediction_keywords(prediction_row)
        if schema_valid:
            schema_valid_count += 1
        duplicate_prediction_items += len(predicted_items) - len(set(predicted_items))
        predicted = set(predicted_items)
        gold = set(gold_keywords)
        true_positive = len(predicted & gold)
        false_positive = predicted - gold
        false_negative = gold - predicted
        total_tp += true_positive
        total_fp += len(false_positive)
        total_fn += len(false_negative)
        precision, recall, f1 = _sample_scores(predicted, gold)
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        if predicted == gold:
            exact_count += 1

        source_text = gold_row.get("normalized_text") or gold_row.get("raw_text") or ""
        hallucinated = sorted(keyword for keyword in predicted if keyword not in source_text)
        predicted_unique_total += len(predicted)
        hallucinated_total += len(hallucinated)
        if hallucinated:
            hallucinated_sample_count += 1

        if predicted != gold or not schema_valid or status != "success":
            error_rows.append({
                "sample_id": sample_id,
                "status": status,
                "schema_valid": schema_valid,
                "schema_errors": schema_errors,
                "source_text": source_text,
                "gold_keywords": sorted(gold),
                "predicted_keywords": sorted(predicted),
                "false_positive": sorted(false_positive),
                "false_negative": sorted(false_negative),
                "hallucinated": hallucinated,
            })

    sample_count = len(gold_by_id)
    micro_precision = _safe_ratio(total_tp, total_tp + total_fp)
    micro_recall = _safe_ratio(total_tp, total_tp + total_fn)
    micro_f1 = _safe_ratio(2 * micro_precision * micro_recall, micro_precision + micro_recall)
    report = {
        "report_version": "keyword-eval-v1",
        "label": "model_evaluation",
        "model_id": model_id,
        "sample_count": sample_count,
        "coverage": {"missing": 0, "extra": 0, "rate": 1.0},
        "status_counts": dict(sorted(status_counts.items())),
        "schema_valid_rate": _safe_ratio(schema_valid_count, sample_count),
        "exact_set_match_rate": _safe_ratio(exact_count, sample_count),
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "true_positive": total_tp,
            "false_positive": total_fp,
            "false_negative": total_fn,
        },
        "macro": {
            "precision": _safe_ratio(macro_precision, sample_count),
            "recall": _safe_ratio(macro_recall, sample_count),
            "f1": _safe_ratio(macro_f1, sample_count),
        },
        "hallucination": {
            "keyword_rate": _safe_ratio(hallucinated_total, predicted_unique_total),
            "sample_rate": _safe_ratio(hallucinated_sample_count, sample_count),
            "keyword_count": hallucinated_total,
        },
        "duplicate_prediction_items": duplicate_prediction_items,
        "error_sample_count": len(error_rows),
    }
    return report, error_rows


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    write_text_atomic(path, content)


def render_markdown(report: Dict[str, Any]) -> str:
    return "\n".join([
        f"# Keyword Evaluation: {report['model_id']}",
        "",
        f"- Samples: {report['sample_count']}",
        f"- Schema valid rate: {report['schema_valid_rate']:.6f}",
        f"- Exact-set match rate: {report['exact_set_match_rate']:.6f}",
        f"- Micro P/R/F1: {report['micro']['precision']:.6f} / {report['micro']['recall']:.6f} / {report['micro']['f1']:.6f}",
        f"- Macro P/R/F1: {report['macro']['precision']:.6f} / {report['macro']['recall']:.6f} / {report['macro']['f1']:.6f}",
        f"- Hallucinated keyword rate: {report['hallucination']['keyword_rate']:.6f}",
        f"- Error samples: {report['error_sample_count']}",
        "",
    ])
