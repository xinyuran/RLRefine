English | [中文](README.md)

# RLRefine

**Refine LLM's Reasoning & Extraction via Reinforcement Learning (SFT → DPO → GRPO)**

> A framework for training small LLMs to perform structured information extraction with systematic reasoning.

> **Note**: This project was primarily designed and tested for **Chinese e-commerce review keyword extraction** by the author (rxy). While the framework architecture is language-agnostic, the built-in prompts and post-processing are optimized for Chinese text. English effectiveness has not been fully validated — users should test and adapt for their own use cases.

---

## 1. The Problem

When using small LLMs (e.g., Qwen2.5-7B-Instruct) for structured extraction tasks, base models have these core failure modes:

| Problem | Manifestation |
|---------|--------------|
| **Shallow reasoning** | Jumps to conclusions, skips critical analysis steps |
| **Hallucination** | Outputs words not present in the original text |
| **Inconsistent granularity** | "large screen" vs "screen" — no unified standard |
| **Format instability** | JSON errors, missing fields, repetitive token loops |
| **Missed negation** | Fails to normalize "不怎么粘黏" (not very sticky) into "不粘黏" (not sticky) |

These issues make it hard to reliably use small models in production extraction pipelines.

---

## 2. The Solution

RLRefine teaches models **how to think** before extracting, using a three-stage training pipeline:

```
Base Model  →  SFT  →  DPO  →  GRPO  →  Refined Model
               (learn   (prefer  (optimize  (systematic
               patterns) quality) accuracy)  reasoning)
```

The core idea: construct training data with high-quality reasoning processes, then use RL to reinforce extraction accuracy. The Schema-driven design ensures the same task definition is used for **both training (reward) and inference (prompt + validation)**.

**How reasoning works**: The prompt template (`prompt_template_3.py`) instructs the model to output a "思考" (Reasoning) section followed by JSON. This reasoning structure is embedded in the prompt itself — the model always receives these instructions. After RL training, the model produces significantly higher-quality reasoning and extraction results.

---

## 3. Results

Chinese e-commerce review used for testing: 终于收到我需要的宝贝了，东西很好，价美物廉，谢谢掌柜的！说实在，这是我购物以来来让我最满意的一次购物。无论是掌柜的态度还是对物品，我都非常满意的。掌柜态度很专业热情，有问必答，回复也很快，我问了不少问题，他都不觉得烦，都会认真回答我，这点我向掌柜表示由衷的敬意，这样的好掌柜可不多。再说宝贝，正是我需要的，收到的时候包装完整，打开后让我惊喜的是，宝贝比我想象中的还要好！不得不得竖起大拇指。下次需要的时候我还会再来的，到时候麻烦掌柜给个优惠哦！

Real comparison on the same review (222 characters) with Qwen2.5-7B-Instruct:

| Dimension | Base Model | RL-Trained Model |
|-----------|-----------|-----------------|
| Inference time | ~8 seconds | ~23 seconds |
| Output tokens | 626 chars | 1,691 chars |
| Keywords extracted | 6 | 8 |
| Reasoning structure | Shallow Markdown list | Systematic 5-step analysis |
| Confidence scores | Strings `"0.95"` (wrong type) | Numbers `0.95` (correct) |
| Hallucination | None in this example | None |
| Missed keywords | "满意", "好", "包装" | None |

**Base model output**: `['宝贝', '态度', '回复', '价美物廉', '购物', '优惠']`

**RL-trained model output**: `['满意', '好', '宝贝', '价美物廉', '态度', '回复', '包装', '优惠']`

The RL-trained model correctly identifies "满意" (satisfied) as the core sentiment word, extracts "包装" (packaging) that the base model missed entirely, and produces properly typed confidence scores.

> **Latency trade-off**: The RL-trained model takes ~3x longer because it generates ~3x more tokens with detailed reasoning. This is expected: the model has learned to think more thoroughly before extracting. See [Performance Notes](#performance-notes) for throughput optimization.

---

## 4. Architecture

The **Schema** is the central organizing principle — defined once, used in both training and inference:

```
┌─────────────────────────────────────────────────────────────┐
│                         Schema                               │
│           (Define task structure in Python once)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┴──────────────┐
         │                            │
         ▼                            ▼
┌─────────────────┐        ┌──────────────────────────┐
│  Training Phase │        │     Inference Phase       │
│                 │        │                           │
│  Reward Builder │        │  Prompt Builder           │
│  ┌────────────┐ │        │  (auto-generate from      │
│  │Accuracy F1 │ │        │   Schema + custom_rules)  │
│  │ (50%)      │ │        │           │               │
│  │Schema Valid│ │        │           ▼               │
│  │ (20%)      │ │        │  LLM Inference (vLLM)     │
│  │Format Check│ │        │           │               │
│  │ (20%)      │ │        │    Parse + Validate       │
│  │Thinking Q. │ │        │    (Schema required fields│
│  │ (10%)      │ │        │     + JSON extraction)    │
│  │Hallucin.   │ │        │           │               │
│  │(-0.1/word) │ │        │    Post-process           │
│  └─────┬──────┘ │        │    (dedup/filter/Top-N)   │
│        │        │        │           │               │
│  SFT→DPO→GRPO  │        │    Output / Jieba Fallback│
│        │        │        └──────────────────────────┘
│        ▼        │
│  Refined Model ─┼──────────────────────▶ (used in inference)
└─────────────────┘
```

---

## 5. Key Ideas

### Schema-Driven Design

The same `TaskSchema` object drives everything:
- **Training**: Reward function uses Schema to compute F1, validate format, penalize hallucination
- **Inference**: PromptBuilder embeds Schema into prompts; processor validates responses against Schema required fields

### Three-Stage RL Training

Each stage addresses a different aspect:
- **SFT**: Teaches *what good reasoning looks like* through demonstrations
- **DPO**: Teaches *preference* by contrasting good vs bad reasoning pairs
- **GRPO**: Directly optimizes *extraction accuracy* (F1) as the primary reward signal

### Reward Design Philosophy

Extraction accuracy (F1) at 50% weight — because **quality is the goal**, format and reasoning constraints are means to that end:

| Dimension | Weight | Purpose |
|-----------|--------|---------|
| Accuracy (F1) | **50%** | Core extraction quality |
| Schema Validation | 20% | Output correctness |
| Format Check | 20% | JSON reliability |
| Thinking Quality | 10% | Reasoning robustness |
| Hallucination Penalty | -0.1/word | Anti-fabrication |

### Prompt-Driven Reasoning

The reasoning process is embedded directly in the prompt template (`prompt_template_3.py`), which instructs the model to output a structured reasoning section ("思考") before the JSON result. This is not a toggleable feature — it's a fundamental part of how the extraction prompt works. The model always receives these instructions; RL training improves the quality of the reasoning it produces.

### Production-Grade Inference Engine

Multi-layer fault tolerance ensures something is always returned:
```
LLM Call → [retry with penalty params] → [Jieba TF-IDF fallback] → Output
```

---

## 6. Quick Start

### Step 1: Start vLLM

```bash
# Option 1: Direct launch
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# Option 2: Use project script
bash scripts/run_vllm.sh 0 8000    # GPU 0, Port 8000
```

Wait for `Application startup complete`.

### Step 2: Configure

```bash
cp .env.example .env
# Edit .env:
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=dummy
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```

### Step 3: Run the Example

```bash
cd RLRefine
python examples/keyword_extraction/run.py
```

Expected output:

```json
{
  "id": 0,
  "data": {
    "keywords": ["屏幕", "电池", "拍照", "夜景"]
  }
}
```

---

## Installation

### Requirements

- Python >= 3.10
- NVIDIA GPU + CUDA >= 12.0

### Setup

```bash
git clone https://github.com/your-username/RLRefine.git
cd RLRefine
pip install -r requirements.txt
pip install python-dotenv
```

> Inference only requires `openai`, `tqdm`, `jieba`. Skip `ms-swift` and `vllm` if you don't need training.

---

## Core API

### Schema — Define Your Task

```python
from core.schema import TaskSchema, FieldDefinition, FieldType, ExtractionTask

schema = TaskSchema(name="keyword_extraction", description="E-commerce keyword extraction")
schema.add_field(FieldDefinition(
    name="keywords",
    type=FieldType.ARRAY_OF_OBJECTS,
    description="Extracted keywords",
    required=True,
))

task = ExtractionTask(
    schema=schema,
    language="zh",
    domain="ecommerce",
    custom_rules=["Keywords must come from original text"]
)
```

### PromptBuilder — Three Methods

```python
from prompts.prompt_builder import PromptBuilder

# Method 1: Built-in battle-tested template (recommended for keyword extraction)
# Uses prompt_template_3.py which includes structured reasoning instructions
prompt_builder = PromptBuilder.create_keyword_extraction_builder()

# Method 2: Auto-generate from Schema (good for custom tasks)
# Note: when using this method, you may want to set
# config.response_format = {"type": "json_object"} to enforce pure JSON output
# prompt_builder = PromptBuilder.from_task(task)

# Method 3: Fully custom
# def my_generator(input_text: str, task) -> tuple:
#     return "You are an extraction expert...", f"Extract: {input_text}"
# prompt_builder = PromptBuilder(custom_prompt_generator=my_generator)
```

> See [`examples/keyword_extraction/run.py`](examples/keyword_extraction/run.py) for full usage.

### Config — Tune Behavior

```python
from core.config import Config

config = Config(task_schema=schema, task=task)
config.max_token = 1024
config.temperature = 0.0
config.max_retries = 3
config.enable_post_process = True
config.post_process_top_n = 8
config.post_process_filter_stopwords = True
config.post_process_filter_not_in_original = True

# response_format defaults to None (allows mixed text output from the prompt template).
# If using Method 2 (Schema-driven) and you want the model to output pure JSON only,
# explicitly set:
# config.response_format = {"type": "json_object"}
```

### RLRefineProcessor — Run Extraction

```python
from core.processor import RLRefineProcessor

processor = RLRefineProcessor(config=config, task=task, prompt_builder=prompt_builder)

# Single item
result = processor.process_single(text="...", text_id="001")

# Batch (multi-threaded, 10 workers by default)
results = processor.process_batch([
    {"id": "001", "describe": "..."},
    {"id": "002", "describe": "..."},
])
```

**Execution flow:**
```
Input → Preprocess → Prompt → LLM → Parse JSON → Schema Validate → Post-process → Output
                                  ↓ on failure
                            Retry (+ penalty params after 2 fails)
                                  ↓ all failed
                            Jieba Fallback → Output
```

---

## RL Training Pipeline

### Training Flow

```
Qwen2.5-7B-Instruct (base)
    │
    ▼ SFT  ── Teach reasoning patterns via demonstration data
    │         (merge LoRA: bash rl/merge_lora.sh)
    │
    ▼ DPO  ── Teach quality preference via good/bad pairs
    │         (merge LoRA: bash rl/merge_lora.sh)
    │
    ▼ GRPO ── Directly optimize extraction F1 as reward
    │         (merge LoRA: bash rl/merge_lora.sh)
    │
    ▼
Refined Model (systematic reasoning + accurate extraction)
```

> After each LoRA training stage, use `rl/merge_lora.sh` to merge the adapter weights into the base model before proceeding to the next stage.

### Step 1: Prepare Training Data

**SFT data**: Use a strong model (e.g., Qwen3-Max) to generate high-quality reasoning + extraction demonstrations. The script `scripts/generate_data_qwen3-max.py` and the prompt template `prompts/prompt_template_3.py` can be used as reference.

The prompt instructs the model to output two sections — a reasoning section ("思考") followed by JSON:

```json
{
  "messages": [
    {"role": "system", "content": "(prompt from prompt_template_3.py)"},
    {"role": "user", "content": "Analyze this review: ..."},
    {"role": "assistant", "content": "**思考**\nStep 1: Identify subjects... 'screen' is a product attribute...\n\n**JSON输出**\n{\"keywords\": [[\"product attribute\", \"screen\", 0.95]]}"}
  ]
}
```

**DPO data**: Use the script `scripts/generate_data_dpo.py` with the prompt `prompts/prompt_template_dpo.py` to generate chosen/rejected preference pairs.

**GRPO data**: Convert SFT data to GRPO format (remove assistant responses, keep only prompts):

```bash
python rl/convert_sft_to_grpo.py
```

> For the ms-swift standard data format requirements for DPO and GRPO training, refer to the [ms-swift custom dataset documentation](https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-dataset.html#dpo-orpo-cpo-simpo-rm).

### Step 2: SFT Training

```bash
bash rl/sft_finetune.sh
bash rl/merge_lora.sh    # Merge LoRA weights into base model
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_PATH` | Qwen2.5-7B-Instruct | Base model |
| `LORA_RANK` | `16` | LoRA rank |
| `LEARNING_RATE` | `1e-5` | Learning rate |
| `MAX_LENGTH` | `8192` | Max sequence length |

### Step 3: DPO Training

```bash
bash rl/dpo_finetune.sh
bash rl/merge_lora.sh    # Merge LoRA weights
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_PATH` | `./output/sft/xxx-merged` | SFT merged model |
| `BETA` | `0.2` | KL penalty coefficient |
| `LEARNING_RATE` | `5e-7` | Learning rate (very small for DPO) |

### Step 4: GRPO Training

```bash
bash rl/grpo_finetune.sh
bash rl/merge_lora.sh    # Merge LoRA weights
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_PATH` | `./output/dpo/xxx-merged` | DPO merged model |
| `BETA` | `0.01` | KL penalty (smaller than DPO) |
| `NUM_GENERATIONS` | `8` | Outputs sampled per prompt |
| `TEMPERATURE` | `0.9` | Sampling temperature |

---

## Custom Tasks

### Example: Sentiment Analysis

```python
# schema.py
def create_sentiment_task() -> ExtractionTask:
    schema = TaskSchema(name="sentiment", description="Text sentiment analysis")
    schema.add_field(FieldDefinition(
        name="sentiment", type=FieldType.STRING, required=True,
        enum_values=["positive", "negative", "neutral"]
    ))
    schema.add_field(FieldDefinition(
        name="confidence", type=FieldType.FLOAT, required=True,
        min_value=0.0, max_value=1.0
    ))
    return ExtractionTask(schema=schema,
                          custom_rules=["Confidence reflects sentiment clarity"])
```

```python
# run.py
task = create_sentiment_task()
config = Config(task_schema=task.schema, task=task)
config.enable_post_process = False
config.response_format = {"type": "json_object"}  # Enforce pure JSON for Schema-driven prompt

builder = PromptBuilder.from_task(task)
processor = RLRefineProcessor(config=config, task=task, prompt_builder=builder)
results = processor.process_batch([{"id": "1", "text": "Great product!"}])
```

---

## Performance Notes

### Single-Item Inference Latency

Single-item inference with vLLM typically takes **5-25 seconds** per item. This is a characteristic of autoregressive LLM inference (tokens generated one at a time), not a framework issue.

| Scenario | Approximate Latency |
|----------|-------------------|
| Short text + base model | ~5 seconds |
| Long text + base model | ~8 seconds |
| Long text + RL-trained model | ~20-25 seconds |

### Throughput Optimization

1. **Use batch processing** — `process_batch()` uses 10 threads by default, dramatically improving throughput for large datasets
2. **Tune vLLM** — Increase `--max-num-seqs` and `--gpu-memory-utilization` in `scripts/run_vllm.sh`
3. **Consider shorter prompts for simple tasks** — The built-in keyword extraction prompt is comprehensive; for simpler tasks, a Schema-driven prompt (Method 2) with `response_format=json_object` produces less output

---

## Directory Structure

```
RLRefine/
├── rl/                                # RL training module
│   ├── reward_builder.py              #   Schema-driven reward function (GRPO)
│   ├── convert_sft_to_grpo.py         #   SFT → GRPO data format converter
│   ├── merge_lora.sh                  #   LoRA weight merging script (after each training stage)
│   ├── sft_finetune.sh                #   SFT training script (ms-swift)
│   ├── dpo_finetune.sh                #   DPO training script (ms-swift)
│   └── grpo_finetune.sh               #   GRPO training script (ms-swift + vLLM)
│
├── core/                              # Core inference module
│   ├── schema.py                      #   Schema definition & validation
│   ├── config.py                      #   Configuration system
│   ├── processor.py                   #   Core processor (inference + retry + fallback)
│   ├── preprocess.py                  #   Text preprocessing
│   ├── post_process.py                #   Keyword post-processing
│   └── fallback.py                    #   Jieba fallback (TF-IDF / TextRank)
│
├── prompts/                           # Prompt module
│   ├── prompt_builder.py              #   Dynamic Prompt generator
│   ├── prompt_template_3.py           #   Keyword extraction template (with reasoning)
│   ├── prompt_template_4_short.py     #   Keyword extraction template (short text)
│   └── prompt_template_dpo.py         #   DPO data generation template (chosen/rejected)
│
├── scripts/                           # Utility scripts
│   ├── run_vllm.sh                    #   vLLM startup script
│   ├── generate_data_qwen3-max.py     #   SFT data generation (reference)
│   └── generate_data_dpo.py           #   DPO data generation (reference)
│
├── examples/
│   └── keyword_extraction/            #   Complete runnable example
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## FAQ

### Q: Can I use a backend other than vLLM?

This project was only tested with **vLLM + Qwen2.5-7B-Instruct**. Other OpenAI-compatible backends should work in theory, but `extra_body={"repetition_penalty": ...}` is vLLM-specific and won't work with OpenAI's official API. The framework has automatic fallback handling for this case (removes `extra_body` and retries).

### Q: Output truncated (finish_reason=length)?

Increase `config.max_token` (default 1024, try 2048).

### Q: Can I use only inference without RL training?

Yes. The `rl/` scripts are optional. You only need `core/`, `prompts/`, and `examples/`. Skip `ms-swift` and `vllm` in installation if not training.

### Q: Why is single-item inference slow?

Autoregressive LLM inference generates one token at a time. RL-trained models are slower because they produce longer, more detailed reasoning. See [Performance Notes](#performance-notes).

### Q: What is `response_format` and when should I set it?

By default, `response_format` is `None`, allowing the model to output mixed text (reasoning + JSON). This is required when using Method 1 (built-in prompt template), since the prompt instructs the model to output a reasoning section before the JSON. If you use Method 2 (Schema-driven prompt) and want to enforce pure JSON output, explicitly set `config.response_format = {"type": "json_object"}`.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM service URL |
| `VLLM_API_KEY` | `dummy` | API key (usually not validated by vLLM) |
| `MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | Model name or local path |
| `DEBUG` | `false` | Enable debug logging |

---

## License

Apache 2.0
