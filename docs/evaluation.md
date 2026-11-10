# Evaluation results and reproducibility

[English overview](../README.md) · [中文说明](../README_CN.md)

These are preserved results from the project's historical experiments, not a
new evaluation performed for this documentation release. The maintainer has
confirmed that **E4** and the public **checkpoint-924** are names for the same
weights. Backend, prompt, decoding and post-processing still affect results.

## Chinese review keywords

The development split contains **331** samples. Macro-F1 averages each sample's
exact-keyword-set F1 equally. Schema validity measures the evaluated output's
field types and structure. It does not mean that every keyword is correct.

| Stage | Dev Macro-F1 | Dev Schema validity |
|---|---:|---:|
| Qwen2.5-7B-Instruct baseline | 0.2490 | 0.4560 |
| SFT | 0.4946 | 0.6737 |
| DPO | 0.5437 | 0.7432 |
| GRPO / E4 | 0.5759 | 0.7976 |
| GRPO / E4 + deterministic contract repair | 0.6932 | 1.0000 |

The final row includes a separate offline repair stage. It is not raw model
performance, and the browser playground does not reproduce that repair stage.
Repair can discard invalid entries and improve contract validity while still
leaving omissions or poor keyword choices. Historical deployment gates were
not passed; these results do not establish production readiness.

The frozen human-reviewed test split contains **496** samples, used once after
candidate selection, without tuning on that split:

| Stage | Test Macro-F1 | Test Micro-F1 | Test Schema validity |
|---|---:|---:|---:|
| DPO | 0.298411 | 0.374718 | 0.473790 |
| GRPO / E4 | 0.352544 | 0.444791 | 0.552419 |
| Difference | +0.054133 | +0.070073 | +0.078629 |

The lower held-out absolute scores are part of the result. Do not substitute
development scores for held-out performance or imply that 100% repaired Schema
validity is 100% task accuracy.

## Support-ticket routing

A separate fixed-field task used 240 human-reviewed synthetic tickets:
120 development, 100 test and 20 challenge examples. The test below used the
base model without the keyword-specific adapter, at concurrency 4.

| Metric on 100 test samples | Unconstrained | Strict JSON Schema |
|---|---:|---:|
| Intent Macro-F1 | 0.8508 | 0.8508 |
| Urgency accuracy | 0.6500 | 0.6500 |
| Evidence exact match | 0.3900 | 0.3900 |
| Schema validity | 1.0000 | 1.0000 |
| Full task-contract validity | 0.9900 | 0.9900 |
| P50 latency | 0.175 s | 0.227 s |
| Throughput | 497.2 tok/s | 368.2 tok/s |

The test outputs were identical across decoding modes. Strict decoding added
about 29.7% to P50 latency and reduced throughput by about 26.0% in this one
experiment. This is an extensibility check on synthetic data, not evidence of
customer-traffic performance or a universal decoding recommendation.

## What you can reproduce from the public release

| Available | Scope |
|---|---|
| Python tests and CPU quickstart | Contract, schema, processor and reward behavior without model calls |
| Static playground and parity tests | Browser/Python agreement on explicit keyword rules |
| Public model and synthetic training data | Model inference and training-data inspection |
| `evaluation.keyword_evaluator` | Exact-set scoring on your own gold/prediction records |

Full frozen evaluation datasets, historical prediction files, all intermediate
checkpoints and experiment manifests are not included in the public release.
The tables therefore document historical results; the public files alone do
not independently reproduce every table. Public training data must not be
relabeled as an unseen test set.

## Use the public scorer

The scorer rejects duplicate IDs and mismatched gold/prediction coverage.
Schema-invalid/error predictions receive an empty predicted set. It reports
Macro/Micro-F1, exact-set match, Schema validity, source-missing keyword rates
and status counts; it does not perform inference or repair.

```python
from evaluation.keyword_evaluator import evaluate_records

gold = [{
    "sample_id": "example-1",
    "normalized_text": "屏幕清晰",
    "gold": {"keywords": ["屏幕", "清晰"]},
}]
predictions = [{
    "sample_id": "example-1",
    "status": "success",
    "data": {"keywords": [["对象", "屏幕", 0.9]]},
}]
report, errors = evaluate_records(gold, predictions, model_id="illustrative-example")
assert abs(report["micro"]["f1"] - 2 / 3) < 1e-9
```

This single hand-written example only demonstrates scoring; it is not a model
benchmark. The scorer's Schema-validity field and the stricter source/uniqueness
contract check answer different questions and should be reported separately.
