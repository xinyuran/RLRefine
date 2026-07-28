"""Paired B1/candidate diagnostics for prompt selection on teacher dev."""
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from evaluation.baseline_diagnostics import (
    _bootstrap_mean_ci,
    _index_unique,
    _prediction_view,
    _sample_f1,
    _top,
    emit,
)
from evaluation.keyword_evaluator import (
    evaluate_records,
    file_sha256,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)


EXPECTED_REFERENCE_SHA256 = "2c4cd0e5eaf14dc4ade0ac857e70d5669d9ea4746608c46c6bf3df6df3d83dfe"
EXPECTED_B1_SHA256 = "deec1c9138efa09910c6cd202f92d2c1bf3979b0a9f09db197bef7a5f4d59abc"
B1_VARIANT = "b1_structured_v3"
B2_VARIANT = "b2_compact_v1"
CANDIDATE_VARIANTS = (B2_VARIANT, "b3_balanced_v1", "sft_v2_json_only_protocol_v1")


def analyze(
    reference_rows: Sequence[Dict[str, Any]],
    b1_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
    max_new_tokens: int,
    bootstrap_iterations: int,
    seed: int,
    noninferiority_margin: float = 0.03,
    candidate_variant: str = B2_VARIANT,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not 0 <= noninferiority_margin < 1:
        raise ValueError("noninferiority margin must be in [0, 1)")
    if candidate_variant not in CANDIDATE_VARIANTS:
        raise ValueError(f"Unsupported candidate variant: {candidate_variant}")
    candidate_tag = candidate_variant.split("_", 1)[0]
    reference = _index_unique(reference_rows, "reference")
    predictions = {
        B1_VARIANT: _index_unique(b1_rows, B1_VARIANT),
        candidate_variant: _index_unique(candidate_rows, candidate_variant),
    }
    for expected_variant, rows in ((B1_VARIANT, b1_rows), (candidate_variant, candidate_rows)):
        if any(row.get("prompt_variant") != expected_variant for row in rows):
            raise ValueError(f"Prediction rows do not match variant {expected_variant}")
    if not reference:
        raise ValueError("Reference rows must not be empty")
    if any(set(reference) != set(rows) for rows in predictions.values()):
        raise ValueError("B1/candidate/reference sample coverage mismatch")

    reports = {
        B1_VARIANT: evaluate_records(reference_rows, b1_rows, B1_VARIANT)[0],
        candidate_variant: evaluate_records(reference_rows, candidate_rows, candidate_variant)[0],
    }
    counters = {
        variant: {
            "issues": Counter(),
            "fp": Counter(),
            "fn": Counter(),
            "hallucinated": Counter(),
            "tokens": [0, 0],
            "max_tokens": 0,
        }
        for variant in predictions
    }
    deltas: List[float] = []
    outcomes = Counter()
    paired_rows = []
    for sample_id in sorted(reference):
        source_text = reference[sample_id].get("normalized_text") or ""
        gold = set(reference[sample_id]["gold"]["keywords"])
        views = {
            variant: _prediction_view(rows[sample_id], source_text)
            for variant, rows in predictions.items()
        }
        scores = {variant: _sample_f1(view["keywords"], gold) for variant, view in views.items()}
        for variant, view in views.items():
            item = counters[variant]
            item["issues"][view["schema_issue"]] += 1
            item["fp"].update(view["keywords"] - gold)
            item["fn"].update(gold - view["keywords"])
            item["hallucinated"].update(view["hallucinated"])
            item["tokens"][0] += view["input_tokens"]
            item["tokens"][1] += view["output_tokens"]
            item["max_tokens"] += int(view["output_tokens"] >= max_new_tokens)
        delta = scores[candidate_variant] - scores[B1_VARIANT]
        deltas.append(delta)
        outcome = f"{candidate_tag}_win" if delta > 0 else "b1_win" if delta < 0 else "tie"
        outcomes[outcome] += 1
        paired_rows.append({
            "sample_id": sample_id,
            "b1_f1": scores[B1_VARIANT],
            f"{candidate_tag}_f1": scores[candidate_variant],
            f"f1_delta_{candidate_tag}_minus_b1": delta,
            "outcome": outcome,
            "b1_schema_issue": views[B1_VARIANT]["schema_issue"],
            f"{candidate_tag}_schema_issue": views[candidate_variant]["schema_issue"],
            "b1_output_tokens": views[B1_VARIANT]["output_tokens"],
            f"{candidate_tag}_output_tokens": views[candidate_variant]["output_tokens"],
        })

    ci_low, ci_high = _bootstrap_mean_ci(deltas, bootstrap_iterations, seed)
    count = len(reference)
    variants = {}
    for variant, item in counters.items():
        metric = reports[variant]
        variants[variant] = {
            "schema_valid_rate": metric["schema_valid_rate"],
            "exact_set_match_rate": metric["exact_set_match_rate"],
            "micro_f1": metric["micro"]["f1"],
            "macro_f1": metric["macro"]["f1"],
            "hallucination_keyword_rate": metric["hallucination"]["keyword_rate"],
            "mean_input_tokens": item["tokens"][0] / count,
            "mean_output_tokens": item["tokens"][1] / count,
            "max_token_output_count": item["max_tokens"],
            "schema_issue_counts": dict(sorted(item["issues"].items())),
            "top_false_positive": _top(item["fp"]),
            "top_false_negative": _top(item["fn"]),
            "top_hallucinated": _top(item["hallucinated"]),
        }
    b1, candidate = variants[B1_VARIANT], variants[candidate_variant]
    input_ratio = candidate["mean_input_tokens"] / b1["mean_input_tokens"]
    output_ratio = candidate["mean_output_tokens"] / b1["mean_output_tokens"]
    is_sft_v2_protocol = candidate_variant == "sft_v2_json_only_protocol_v1"
    thresholds = {
        "macro_f1_noninferiority_margin": noninferiority_margin,
        "max_input_token_ratio": 1.05 if is_sft_v2_protocol else 0.5,
        "max_schema_valid_rate_drop": 0.02,
        "max_hallucination_rate_increase": 0.01,
    }
    checks = {
        f"{candidate_tag}_macro_f1_noninferior": ci_low >= -noninferiority_margin,
        (
            f"{candidate_tag}_input_tokens_within_five_percent"
            if is_sft_v2_protocol else f"{candidate_tag}_input_tokens_at_most_half"
        ): input_ratio <= thresholds["max_input_token_ratio"],
        f"{candidate_tag}_output_tokens_lower": output_ratio < 1,
        f"{candidate_tag}_no_max_token_outputs": candidate["max_token_output_count"] == 0,
        f"{candidate_tag}_schema_valid_rate_within_tolerance": candidate["schema_valid_rate"] >= b1["schema_valid_rate"] - thresholds["max_schema_valid_rate_drop"],
        f"{candidate_tag}_hallucination_rate_within_tolerance": candidate["hallucination_keyword_rate"] <= b1["hallucination_keyword_rate"] + thresholds["max_hallucination_rate_increase"],
    }
    report = {
        "report_version": f"prompt-optimization-b1-{candidate_tag}-v1",
        "label": "teacher_dev_prompt_optimization_not_human_gold_test",
        "sample_count": count,
        "bootstrap_iterations": bootstrap_iterations,
        "seed": seed,
        "thresholds": thresholds,
        "paired": {
            f"macro_f1_delta_{candidate_tag}_minus_b1": sum(deltas) / count,
            "bootstrap_95_ci": [ci_low, ci_high],
            "outcomes": dict(sorted(outcomes.items())),
        },
        f"token_cost_ratio_{candidate_tag}_over_b1": {"input": input_ratio, "output": output_ratio},
        "variants": variants,
        "checks": checks,
        "candidate_variant": candidate_variant,
        "provisional_decision": (
            f"freeze_{candidate_tag}_prompt"
            if all(checks.values())
            else f"retain_b1_or_iterate_{candidate_tag}"
        ),
    }
    return report, paired_rows


def render_markdown(report: Dict[str, Any]) -> str:
    paired = report["paired"]
    candidate_variant = report["candidate_variant"]
    candidate_tag = candidate_variant.split("_", 1)[0]
    b1 = report["variants"][B1_VARIANT]
    candidate = report["variants"][candidate_variant]
    token_ratio = report[f"token_cost_ratio_{candidate_tag}_over_b1"]
    return "\n".join([
        f"# Paired B1/{candidate_tag.upper()} Prompt Optimization Diagnostics",
        "",
        f"- Label: `{report['label']}`",
        f"- Samples: {report['sample_count']}",
        f"- Macro-F1 delta ({candidate_tag.upper()}-B1): {paired[f'macro_f1_delta_{candidate_tag}_minus_b1']:.6f}",
        f"- Bootstrap 95% CI: [{paired['bootstrap_95_ci'][0]:.6f}, {paired['bootstrap_95_ci'][1]:.6f}]",
        f"- B1/{candidate_tag.upper()} micro-F1: {b1['micro_f1']:.6f} / {candidate['micro_f1']:.6f}",
        f"- {candidate_tag.upper()}/B1 input/output token ratio: {token_ratio['input']:.3f} / {token_ratio['output']:.3f}",
        f"- Max-token outputs B1/{candidate_tag.upper()}: {b1['max_token_output_count']} / {candidate['max_token_output_count']}",
        f"- Decision: `{report['provisional_decision']}`",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=Path("reports/baselines/qwen2_5_7b_dev/v1"))
    parser.add_argument("--candidate-dir", "--b2-dir", dest="candidate_dir", type=Path, default=Path("reports/baselines/qwen2_5_7b_dev/v2"))
    parser.add_argument("--candidate-variant", choices=CANDIDATE_VARIANTS, default=B2_VARIANT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noninferiority-margin", type=float, default=0.03)
    args = parser.parse_args()
    try:
        candidate_tag = args.candidate_variant.split("_", 1)[0]
        output_dir = args.output_dir or args.candidate_dir / "diagnostics"
        paths = {
            "reference": args.baseline_dir / "dev_teacher_reference.jsonl",
            "b1": args.baseline_dir / "b1_structured_v3_predictions.jsonl",
            candidate_tag: args.candidate_dir / f"{args.candidate_variant}_predictions.jsonl",
        }
        if file_sha256(paths["reference"]) != EXPECTED_REFERENCE_SHA256:
            raise ValueError("Dev reference hash mismatch")
        if file_sha256(paths["b1"]) != EXPECTED_B1_SHA256:
            raise ValueError("B1 prediction hash mismatch")
        candidate_reference = args.candidate_dir / "dev_teacher_reference.jsonl"
        if file_sha256(candidate_reference) != EXPECTED_REFERENCE_SHA256:
            raise ValueError("Candidate dev reference hash mismatch")
        baseline_manifest = json.loads((args.baseline_dir / "run_manifest.json").read_text(encoding="utf-8"))
        candidate_manifest = json.loads((args.candidate_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if candidate_manifest.get("variants") != [args.candidate_variant]:
            raise ValueError(f"Candidate manifest must contain only {args.candidate_variant}")
        controlled_fields = (
            "model_path", "dev_sha256", "dev_sample_ids_sha256", "batch_size",
            "max_input_tokens", "max_new_tokens", "do_sample", "seed",
        )
        mismatches = {
            field: [baseline_manifest.get(field), candidate_manifest.get(field)]
            for field in controlled_fields
            if baseline_manifest.get(field) != candidate_manifest.get(field)
        }
        if mismatches:
            raise ValueError(f"B1/candidate controlled configuration mismatch: {mismatches}")
        report, rows = analyze(
            read_jsonl(paths["reference"]), read_jsonl(paths["b1"]), read_jsonl(paths[candidate_tag]),
            candidate_manifest["max_new_tokens"], args.bootstrap_iterations, args.seed,
            args.noninferiority_margin, args.candidate_variant,
        )
        report["input_hashes"] = {name: file_sha256(path) for name, path in paths.items()}
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / f"paired_b1_{candidate_tag}_diagnostics.json"
        markdown_file = output_dir / f"paired_b1_{candidate_tag}_diagnostics.md"
        samples_file = output_dir / f"paired_b1_{candidate_tag}_samples.jsonl"
        write_json_atomic(report_file, report)
        write_text_atomic(markdown_file, render_markdown(report))
        write_jsonl_atomic(samples_file, rows)
        emit(
            "p1_prompt_optimization_complete", status="PASS",
            report_file=str(report_file), report_sha256=file_sha256(report_file),
            markdown_file=str(markdown_file), markdown_sha256=file_sha256(markdown_file),
            samples_file=str(samples_file), samples_sha256=file_sha256(samples_file),
            paired=report["paired"],
            **{f"token_cost_ratio_{candidate_tag}_over_b1": report[f"token_cost_ratio_{candidate_tag}_over_b1"]},
            checks=report["checks"], provisional_decision=report["provisional_decision"],
            variant_diagnostics={name: {
                "schema_issue_counts": values["schema_issue_counts"],
                "max_token_output_count": values["max_token_output_count"],
                "top_false_positive": values["top_false_positive"][:10],
                "top_false_negative": values["top_false_negative"][:10],
                "top_hallucinated": values["top_hallucinated"][:10],
            } for name, values in report["variants"].items()},
        )
        return 0
    except Exception as exc:
        emit("p1_prompt_optimization_complete", status="FAIL", error_type=type(exc).__name__, error=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
