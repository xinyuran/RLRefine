# StructAlign

English | [简体中文](README.md)

A schema-driven toolkit for structured information extraction with large
language models.

StructAlign aligns task definitions, inference constraints, post-training
rewards, and output validation around a shared schema and semantic contract.
Its primary example extracts keywords from Chinese e-commerce reviews. A
support-ticket routing example demonstrates how the same abstractions can
support a different output structure.

## Highlights

- **Schema-driven tasks** — define field types, enums, lengths, and nested
  structures in one place.
- **Training–inference alignment** — SFT, DPO, GRPO, and inference share the
  same structured target.
- **Semantic contracts** — validate source faithfulness, keyword length,
  cardinality, and duplicates beyond basic JSON Schema checks.
- **Reliability controls** — distinguish retries, deterministic repairs,
  fallback extraction, and explicit failure states.
- **OpenAI-compatible inference** — connect to vLLM and other compatible
  serving endpoints.
- **Extensible task shapes** — support both keyword lists and fixed-field
  classification through the same core abstractions.

## How It Works

```text
Task schema
    │
    ├── Prompt and JSON Schema
    ├── SFT / DPO / GRPO targets
    └── Inference validation
             │
             ├── Valid output
             ├── Deterministically repairable output
             └── Failure or human-review candidate
```

StructAlign keeps three questions separate:

1. Did the request complete successfully?
2. Does the output satisfy its JSON Schema?
3. Does it satisfy task semantics, such as requiring every keyword to appear
   in the source text?

This separation prevents syntactically valid JSON from being mistaken for a
trustworthy task result.

## Quickstart

The core setup requires Python 3.10+ and does not need a GPU, model weights, or
an external API:

```bash
git clone https://github.com/xinyuran/RLRefine.git
cd RLRefine
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m scripts.contract_quickstart
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The CPU-only quickstart validates the keyword-extraction and ticket-routing
schemas:

```json
{
  "keyword_contract": {"valid": true, "errors": []},
  "routing_schema": {"valid": true, "errors": []}
}
```

Run the public test suite:

```bash
python -m unittest discover -v tests
```

## Model Serving

Install the optional GPU training and serving stack:

```bash
python -m pip install -r requirements-training.txt
```

Start an OpenAI-compatible vLLM server:

```bash
bash scripts/run_vllm.sh 0 8001 /path/to/model
```

Configure and run the keyword-extraction example:

```bash
cp examples/keyword_extraction/.env.example examples/keyword_extraction/.env
python examples/keyword_extraction/run.py
```

The main inference entry point is `core.processor.RLRefineProcessor`. Example
task schemas are available in `examples/keyword_extraction/schema.py` and
`examples/intent_routing/schema.py`.

## Post-training Setup

The validated pipeline starts from Qwen2.5-7B-Instruct and applies LoRA-based
supervised fine-tuning, preference optimization, and reward-driven
reinforcement learning.

| Stage | Primary objective | LoRA rank / alpha | Learning rate | Epochs | Key setting |
|---|---|---:|---:|---:|---|
| SFT | Learn the task and output format | 16 / 64 | `1e-5` | 5 | Maximum length 8192 |
| DPO | Learn preferences between better and worse answers | 8 / 32 | `5e-7` | 1 | `beta=0.2` |
| GRPO | Directly optimize extraction and structure rewards | 8 / 16 | `5e-7` | 1 | `beta=0.01`, 4 generations per prompt |

Main training environment:

| Item | Configuration |
|---|---|
| GPUs | 2 × NVIDIA H100 80GB |
| Precision | BF16 |
| Training framework | ms-swift 3.11.2 |
| Inference and generation | vLLM 0.13.0 |
| Parallelism and memory | DeepSpeed ZeRO-2, colocated vLLM |
| Base model | Qwen2.5-7B-Instruct |

Training entry points are located in `rl/`. Their paths are examples; users
must provide their own model and datasets.

## Evaluation Results

The following results were measured offline on a fixed development set.
**Macro-F1** gives each sample's F1 equal weight, while **schema validity**
measures the percentage of outputs with the required fields, types, and
structure.

| Model stage | Macro-F1 | Schema validity |
|---|---:|---:|
| Base model | 0.2490 | 0.4560 |
| SFT | 0.4946 | 0.6737 |
| DPO | 0.5437 | 0.7432 |
| GRPO | 0.5759 | 0.7976 |
| GRPO + deterministic contract repair | 0.6932 | 1.0000 |

On a separate set of 496 manually reviewed examples, GRPO improved Macro-F1,
Micro-F1, and schema validity over DPO by `0.0541`, `0.0701`, and `0.0786`,
respectively. This held-out set was not used for training or model selection.

Schema portability was also evaluated on a fixed-field routing task built from
240 manually reviewed synthetic support tickets. On its 100 held-out test
examples, unconstrained and strict JSON Schema decoding produced identical
quality metrics:

| Metric | Unconstrained | Strict JSON Schema |
|---|---:|---:|
| Intent Macro-F1 | 0.8508 | 0.8508 |
| Urgency accuracy | 0.6500 | 0.6500 |
| Schema validity | 1.0000 | 1.0000 |
| Full semantic-contract validity | 0.9900 | 0.9900 |
| P50 latency | 0.175 s | 0.227 s |
| Throughput | 497.2 tok/s | 368.2 tok/s |

For this task, strict decoding did not improve quality, while P50 latency
increased by about `29.7%` and throughput decreased by about `26.0%`. Whether
constrained decoding should be enabled therefore depends on the observed error
distribution and performance budget.

## Repository Layout

```text
core/        Schemas, configuration, inference, processing, and contracts
prompts/     Prompt construction for keyword extraction
rl/          SFT, DPO, GRPO, LoRA merging, and reward implementations
examples/    Keyword extraction, ticket routing, and small format examples
scripts/     CPU quickstart and vLLM launcher
tests/       Core schema, inference, and reward tests
```

## Limitations

- Reported quality results primarily cover Chinese e-commerce review keyword
  extraction.
- The support-ticket dataset is small and template-generated; it does not
  represent real customer-service traffic.
- Deterministic repair only covers errors with explicit rules and does not
  replace human review.
- Offline metrics and single-host serving benchmarks are not production SLOs.
- Model weights, training datasets, internal evaluation reports, and production
  deployment configuration are not included.

## License

[MIT License](LICENSE)
