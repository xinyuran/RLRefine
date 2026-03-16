[English](README.md) | [中文](README_CN.md)

# RLRefine

**Refine LLM's Reasoning & Extraction via Reinforcement Learning**

> SFT → DPO → GRPO training pipeline for structured information extraction

> **Note**: This project was primarily designed and tested for **Chinese e-commerce review keyword extraction** by the author (rxy). While the framework architecture is language-agnostic, the built-in prompts and post-processing are optimized for Chinese text. English effectiveness has not been fully validated — users should test and adapt for their own use cases.

---

## What Problem Does This Solve?

When using LLMs for keyword extraction and other information extraction tasks, base models have these core issues:

- **Shallow reasoning**: Models lack systematic analysis steps, often skipping critical reasoning, leading to inconsistent extraction quality
- **Poor keyword quality**: Missing important keywords, inconsistent granularity, inability to handle negation scenarios
- **Hallucination**: Outputting keywords not present in the original text
- **Uncontrollable output format**: JSON format errors, missing fields, repetitive token loops

**RLRefine's approach**: Construct training data with high-quality reasoning processes, use a `SFT → DPO → GRPO` three-stage RL training pipeline to teach models better reasoning patterns, thereby **improving extraction accuracy and consistency**. JSON output stability is an engineering benefit gained during training.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Full RL Training Pipeline** | Built-in SFT → DPO → GRPO training scripts, progressively refining model reasoning and extraction (based on ms-swift) |
| **Schema-Driven Reward** | GRPO reward function is **dominated by extraction accuracy (F1)** (50% weight), also considering reasoning quality, format correctness, and hallucination penalty |
| **Schema-Driven** | Define JSON Schema in Python code; the framework auto-generates Prompts, validates responses, and builds Rewards |
| **Dynamic Prompt Generation** | Auto-generates System/User Prompts based on Schema and task rules, with automatic long/short text switching |
| **Multi-Level Fault Tolerance** | LLM call → retry with penalty params → Jieba rule-based fallback → hard truncation fallback, ensuring production environments always get a return value |
| **Complete Post-Processing Pipeline** | Deduplication, Top-N selection, stopword filtering, time/date word filtering, original text alignment validation, smart backfill |
| **Ready-to-Use Examples** | Includes complete keyword extraction example that can be run directly |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      RLRefine Framework                       │
├────────────────────────────┬─────────────────────────────────┤
│                            │                                  │
│   RL Training Pipeline     │   Inference Engine               │
│   (Core)                   │   (Engineering)                  │
│                            │                                  │
│  ┌─────┐  ┌─────┐  ┌─────┐│  ┌──────────┐  ┌────────────┐  │
│  │ SFT │─▶│ DPO │─▶│GRPO ││  │  Schema   │─▶│  Prompt    │  │
│  └─────┘  └─────┘  └──┬──┘│  │  (Define) │  │  Builder   │  │
│                        │   │  └──────────┘  └──────┬─────┘  │
│  ┌─────────────────────▼─┐ │                       │         │
│  │  Schema-Based Reward  │ │  ┌────────────────────▼──────┐  │
│  │                       │ │  │     LLM (vLLM) Inference  │  │
│  │  Accuracy F1  (50%)   │ │  └────────────────────┬──────┘  │
│  │  Schema Valid (20%)   │ │                       │         │
│  │  Format Check (20%)   │ │         ┌─────────────▼───────┐ │
│  │  Thinking Q.  (10%)   │ │         │  _parse_response()  │ │
│  │  Hallucin.  (-0.1/w)  │ │         │  Schema Validate    │ │
│  └───────────────────────┘ │         └─────────┬───────────┘ │
│                            │                   │              │
│                            │      ┌────────────▼──────────┐  │
│                            │      │   post_process()       │  │
│                            │      │   Dedup/Filter/Top-N   │  │
│                            │      └────────────┬──────────┘  │
│                            │                   │ On failure   │
│                            │      ┌────────────▼──────────┐  │
│                            │      │   Jieba Fallback       │  │
│                            │      └───────────────────────┘  │
└────────────────────────────┴─────────────────────────────────┘
```

---

## Directory Structure

```
RLRefine/
├── rl/                            # Reinforcement learning training module (core)
│   ├── reward_builder.py          #   Schema-driven reward function (for GRPO)
│   ├── convert_sft_to_grpo.py     #   SFT to GRPO data format converter
│   ├── sft_finetune.sh            #   SFT training script (based on ms-swift)
│   ├── dpo_finetune.sh            #   DPO training script (based on ms-swift)
│   └── grpo_finetune.sh           #   GRPO training script (based on ms-swift + vLLM)
│
├── core/                          # Core module
│   ├── schema.py                  #   Schema definition & validation (TaskSchema, FieldDefinition, ExtractionTask)
│   ├── config.py                  #   Configuration system (supports .env / YAML / dict)
│   ├── processor.py               #   Core processor (inference + retry + fallback + post-processing)
│   ├── preprocess.py              #   Text preprocessing (noise removal, emoji, URL, etc.)
│   ├── post_process.py            #   Keyword post-processing (dedup, filter, Top-N, original text validation)
│   └── fallback.py                #   Jieba fallback extraction (TF-IDF / TextRank / simple segmentation)
│
├── prompts/                       # Prompt module
│   ├── prompt_builder.py          #   Dynamic Prompt generator (auto-builds from Schema)
│   ├── prompt_template_3.py       #   Keyword extraction Prompt template (long text)
│   └── prompt_template_4_short.py #   Keyword extraction Prompt template (short text)
│
├── scripts/                       # Utility scripts
│   └── run_vllm.sh                #   One-click vLLM inference service startup
│
├── examples/                      # Usage examples
│   └── keyword_extraction/        #   Keyword extraction example
│       ├── schema.py              #     Task Schema definition
│       ├── config.py              #     Task configuration
│       ├── run.py                 #     Entry point
│       ├── sample_data.jsonl      #     Sample data
│       └── .env.example           #     Environment variable template
│
├── .env.example                   # Environment variable template
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Installation

### Requirements

- Python >= 3.10
- NVIDIA GPU (required for both inference and training)
- CUDA >= 12.0

### Setup

```bash
# 1. Clone the project
git clone https://github.com/your-username/RLRefine.git
cd RLRefine

# 2. Install dependencies
pip install -r requirements.txt
pip install python-dotenv

# 3. Configure environment variables
cp .env.example .env
# Edit .env, modify VLLM_BASE_URL and MODEL_NAME
```

`requirements.txt` contains:

```
openai>=1.0.0        # OpenAI-compatible client (for calling vLLM)
tqdm>=4.66.0         # Progress bar
jieba>=0.42.1        # Chinese word segmentation (for fallback extraction)
pyyaml>=6.0.0        # YAML configuration support
ms-swift>=3.11.0     # RL training framework (only needed for training)
vllm>=0.6.0          # Inference backend (only needed server-side)
```

> If you only need inference (no training), you can skip installing `ms-swift` and `vllm`. Inference only requires `openai`, `tqdm`, and `jieba`.

---

## RL Training Pipeline

This is the core contribution of this project. Through three-stage progressive training, the model's reasoning process and extraction capabilities are refined step by step.

### Training Flow Overview

```
Base Model (Qwen2.5-7B-Instruct)
    │
    ▼
 SFT Training ──── Inject high-quality reasoning demonstrations, teach model "how to think"
    │
    ▼
 DPO Training ──── Through good/bad comparison, teach model to distinguish good vs bad reasoning
    │
    ▼
 GRPO Training ─── Directly optimize model behavior with extraction accuracy as primary reward
    │
    ▼
 Refined Model ─── More systematic thinking, more accurate extraction, more stable output
```

### Why RL?

Typical issues with base LLMs in keyword extraction:

1. **Insufficient reasoning**: Jumping to conclusions without step-by-step text analysis
2. **Missing keywords**: Failing to recognize negation scenarios (e.g., "no sound" should extract "sound")
3. **Inconsistent granularity**: Sometimes outputting "large screen", sometimes "screen", lacking unified standards
4. **Hallucination**: Outputting words not present in the original text

Through RL training, models learn to:
- Systematically decompose text, identifying subject words before descriptors
- Maintain strict original text alignment awareness, never fabricating non-existent words
- Achieve consistent keyword granularity (atomic keywords, ≤4 Chinese characters)
- Produce standardized output format (JSON stability as an accompanying benefit)

### Step 1: Prepare SFT Data

SFT data is the foundation of the entire training pipeline. You need to construct demonstration data with **high-quality reasoning processes** in JSONL format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a keyword extraction expert..."},
    {"role": "user", "content": "Please analyze the following review: This phone has a large screen..."},
    {"role": "assistant", "content": "<think>Analyzing review content: 1. 'large screen' → subject 'screen', descriptor 'large'; 2. 'durable battery' → subject 'battery'...</think>\n{\"keywords\": [[\"Identified subject 'screen'\", \"screen\", 0.95], [\"Identified descriptor 'large'\", \"large\", 0.90]]}"}
  ]
}
```

> Key: The assistant's response must include a structured reasoning process (`<think>...</think>` tags) — this is the core content the model needs to learn.

### Step 2: SFT Training

Use ms-swift for supervised fine-tuning to teach the model the basic patterns of reasoning and extraction:

```bash
# Edit paths in rl/sft_finetune.sh, then run
bash rl/sft_finetune.sh
```

Key parameters in `sft_finetune.sh`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_PATH` | Qwen2.5-7B-Instruct | Base model path |
| `DATASET_PATH` | `sft_data.jsonl` | SFT training data path |
| `LORA_RANK` | `16` | LoRA rank |
| `LORA_ALPHA` | `64` | LoRA Alpha |
| `LEARNING_RATE` | `1e-5` | Learning rate |
| `MAX_LENGTH` | `8192` | Maximum sequence length |

### Step 3: DPO Training

DPO requires preference data pairs (chosen / rejected). You can use a stronger model (e.g., Qwen-Max) to auto-generate these.

```bash
# Edit paths in rl/dpo_finetune.sh, then run
bash rl/dpo_finetune.sh
```

Key parameters in `dpo_finetune.sh`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_PATH` | `./output/sft_merged_model` | SFT merged model path |
| `DATASET_PATH` | `./data/dpo_dataset.jsonl` | DPO training data path |
| `BETA` | `0.2` | KL divergence penalty coefficient |
| `LEARNING_RATE` | `5e-7` | Learning rate (DPO needs very small LR) |
| `LORA_RANK` | `8` | LoRA rank |

### Step 4: GRPO Training

GRPO is the most critical step. It directly optimizes model behavior with extraction accuracy as the objective through **sampling multiple outputs → computing rewards → policy optimization**.

```bash
# 1. Convert SFT data to GRPO format (remove assistant replies, keep only prompts)
python rl/convert_sft_to_grpo.py

# 2. Edit paths in rl/grpo_finetune.sh, then run
bash rl/grpo_finetune.sh
```

Key parameters in `grpo_finetune.sh`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_PATH` | `./output/dpo/checkpoint-xxx-merged` | DPO merged model path |
| `BETA` | `0.01` | KL penalty coefficient (GRPO needs smaller value) |
| `NUM_GENERATIONS` | `8` | Number of sampled outputs per prompt |
| `TEMPERATURE` | `0.9` | Sampling temperature (needs to be high enough for diversity) |
| `REWARD_FUNCS` | `schema_based_reward` | Reward function name |
| `EXTERNAL_PLUGINS` | `reward_builder.py` | Reward function file |

### Reward Function

`SchemaBasedReward` in `rl/reward_builder.py` evaluates model outputs across multiple dimensions, **dominated by extraction accuracy**:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **Accuracy** | **0.5** | **F1 Score against reference answer (core metric)** |
| **Schema Validation** | 0.2 | Required fields present, keyword item format correct |
| **Format Check** | 0.2 | JSON validity, tag completeness |
| **Thinking Quality** | 0.1 | Reasoning process length appropriate, contains analytical vocabulary |
| **Hallucination Penalty** | -0.1/word | Deduction for keywords not in original text (max -0.3) |

> Design philosophy: Accuracy accounts for 50% because **extraction quality is the ultimate goal**. Format and reasoning constraints are means to serve accuracy, not ends in themselves.

---

## Quick Start

### Step 1: Start vLLM Inference Service

RLRefine uses vLLM as the inference backend (via OpenAI-compatible API).

```bash
# Option 1: Direct launch
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# Option 2: Use project script (supports GPU and port args)
bash scripts/run_vllm.sh 0 8000    # GPU 0, Port 8000
```

Wait until you see `Application startup complete`, then the service is ready.

### Step 2: Configure Environment Variables

Edit the `.env` file (or `examples/keyword_extraction/.env`):

```bash
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=dummy
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```

### Step 3: Run the Example

```bash
cd RLRefine
python examples/keyword_extraction/run.py
```

You will see output similar to:

```json
{
  "id": 0,
  "data": {
    "keywords": ["屏幕", "电池", "拍照", "夜景"]
  }
}
```

---

## Core Concepts

### 1. Schema — Define Your JSON Structure

`TaskSchema` is the starting point of the entire framework. You use it to define the expected JSON output structure from the LLM. The framework will automatically:
- Embed the Schema in the Prompt, telling the model what format to output
- Validate LLM responses using the Schema's required fields
- Build GRPO Reward functions using the Schema

```python
from core.schema import TaskSchema, FieldDefinition, FieldType, ExtractionTask

# Define Schema
schema = TaskSchema(
    name="keyword_extraction",
    description="E-commerce review keyword extraction",
)

# Add fields
schema.add_field(FieldDefinition(
    name="keywords",
    type=FieldType.ARRAY,
    description="Extracted keyword list",
    required=True,              # Required field: must be present in LLM response
    min_length=1,               # Extract at least 1 keyword
    max_length=15               # Extract at most 15 keywords
))

schema.add_field(FieldDefinition(
    name="category",
    type=FieldType.STRING,
    description="Review category",
    required=False              # Optional field
))

# Create task
task = ExtractionTask(
    schema=schema,
    task_type="extraction",
    language="zh",
    domain="ecommerce",
    enable_thinking=False,      # Only set True for RL-trained models (see FAQ)
    custom_rules=[              # Task-specific rules (embedded in Prompt)
        "Keyword length limited to 1-4 Chinese characters",
        "Must come from original text, no fabrication"
    ]
)
```

### 2. Config — Control Framework Behavior

The `Config` class manages all configurable options. You can load configuration from `.env` files, YAML files, or dictionaries.

```python
from core.config import Config

config = Config(task_schema=schema, task=task)

# Inference parameters
config.max_token = 1024         # Max generation tokens
config.temperature = 0.0        # Temperature (0 = deterministic output)
config.max_retries = 3          # Max retry count
config.seed = 42                # Random seed

# Penalty parameters (auto-enabled after 2 retries, breaks repetition loops)
config.frequency_penalty = 0.3
config.repetition_penalty = 1.1

# Post-processing switches
config.enable_post_process = True
config.post_process_top_n = 8                    # Keep top N keywords
config.post_process_filter_stopwords = True      # Filter stopwords
config.post_process_filter_time = True           # Filter time words (e.g., "8 o'clock")
config.post_process_filter_date = True           # Filter date words (e.g., "27th")
config.post_process_filter_long = True           # Filter overly long keywords
config.post_process_max_keyword_length = 6       # Max keyword length
config.post_process_filter_not_in_original = True # Filter keywords not in original text
```

### 3. PromptBuilder — Auto-Generate Prompts

`PromptBuilder` automatically constructs System Prompt and User Prompt based on Schema and task rules.

Three usage methods are available — you can flexibly switch between them (just comment/uncomment the corresponding code):

```python
from prompts.prompt_builder import PromptBuilder

# --- Method 1: Use built-in keyword extraction Prompt (battle-tested template) ---
# Suitable for direct use in keyword extraction tasks, no need to write Prompts manually
prompt_builder = PromptBuilder.create_keyword_extraction_builder()

# --- Method 2: Auto-generate from Task (Schema-driven) ---
# Framework auto-generates Prompt based on your defined Schema and custom_rules
# prompt_builder = PromptBuilder.from_task(task)

# --- Method 3: Fully custom ---
# Use when you need full control over Prompt content
# def my_prompt_generator(input_text: str, task) -> tuple:
#     system = "You are an extraction expert..."
#     user = f"Extract: {input_text}"
#     return system, user
# prompt_builder = PromptBuilder(custom_prompt_generator=my_prompt_generator)
```

> See [`examples/keyword_extraction/run.py`](examples/keyword_extraction/run.py) for complete usage of all three methods.

### 4. RLRefineProcessor — Core Processor

Combines the above components to execute the complete inference flow:

```python
from core.processor import RLRefineProcessor

processor = RLRefineProcessor(
    config=config,
    task=task,
    prompt_builder=builder
)

# Single text processing
result = processor.process_single(
    text="This phone has a large screen and durable battery, but camera is mediocre",
    text_id="001"
)
print(result)
# {'id': '001', 'data': {'keywords': ['screen', 'battery', 'camera']}}

# Batch processing (automatic multi-threaded concurrency)
input_data = [
    {"id": "001", "describe": "This phone has a large screen"},
    {"id": "002", "describe": "Clothing quality is great, fabric is comfortable"},
    {"id": "003", "describe": "Fast logistics, intact packaging"},
]
results = processor.process_batch(input_data)
```

**Complete execution flow of `process_single`:**

```
Input Text
  │
  ▼
Preprocessing (noise removal, emoji, URL, truncation)
  │
  ▼
Generate Prompt (PromptBuilder builds from Schema)
  │
  ▼
Call LLM (via vLLM OpenAI-compatible API)
  │
  ├─ Success ──▶ Parse JSON ──▶ Schema Validation ──▶ Post-Processing ──▶ Return Result
  │
  ├─ Failure ──▶ Retry (first 2 with normal params, next 2 with penalty params)
  │
  └─ All Failed ──▶ Jieba Fallback Extraction ──▶ Return Fallback Result
```

---

## Inference Engine Fault Tolerance

In production environments, LLM inference cannot succeed 100% of the time — robust fault tolerance is essential. This is the framework's engineering capability.

### Layer 1: Retry with Penalty Parameters

When LLM output JSON parsing fails, the framework auto-retries. The first 2 attempts use normal parameters; from the 3rd attempt, `frequency_penalty` and `repetition_penalty` are added to break potential repetitive token loops.

```python
# Core logic in processor.py
while retry_count <= max_retries and parsed_data is None:
    use_penalty = retry_count >= 2  # Add penalty from 3rd attempt
    response = self._call_llm(system_prompt, user_prompt, use_penalty, retry_count)
    parsed_data = self._parse_response(response)
    retry_count += 1
```

### Layer 2: Jieba Fallback Extraction

When LLM completely fails, degrade to rule-based Jieba extraction:

- **TF-IDF** (default): Extract keywords based on term frequency-inverse document frequency
- **TextRank**: Extract keywords using graph algorithms
- **Simple segmentation**: Extract by POS tags (nouns, adjectives)

### Layer 3: Response Parsing Tolerance

The `_parse_response` method does not rely on the LLM returning perfect pure JSON. Instead, it:

1. Uses regex `re.search(r'\{.*\}', response)` to extract JSON fragments from any text
2. If Thinking mode is enabled, first strips `<tag>...</tag>` tags
3. Validates based on Schema's required fields, rather than hardcoded field name checks

---

## Custom Tasks

Below are the complete steps to create a new task (e.g., sentiment analysis).

### 1. Create Task Directory

```
examples/
└── sentiment_analysis/
    ├── schema.py
    ├── config.py
    ├── run.py
    └── .env
```

### 2. Define Schema (`schema.py`)

```python
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from core.schema import TaskSchema, FieldDefinition, FieldType, ExtractionTask

def create_sentiment_task() -> ExtractionTask:
    schema = TaskSchema(name="sentiment", description="Text sentiment analysis")

    schema.add_field(FieldDefinition(
        name="sentiment",
        type=FieldType.STRING,
        description="Sentiment polarity",
        required=True,
        enum_values=["positive", "negative", "neutral"]
    ))

    schema.add_field(FieldDefinition(
        name="confidence",
        type=FieldType.FLOAT,
        description="Confidence score",
        required=True,
        min_value=0.0,
        max_value=1.0
    ))

    schema.add_field(FieldDefinition(
        name="aspects",
        type=FieldType.ARRAY_OF_OBJECTS,
        description="Aspect-level sentiment",
        required=False
    ))

    return ExtractionTask(
        schema=schema,
        task_type="extraction",
        language="zh",
        domain="ecommerce",
        enable_thinking=False,
        custom_rules=[
            "Analyze both positive and negative sentiments",
            "Confidence reflects the clarity of sentiment"
        ]
    )
```

### 3. Define Config (`config.py`)

```python
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from core.config import Config
from schema import create_sentiment_task

class SentimentConfig(Config):
    def __init__(self):
        task = create_sentiment_task()
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        if not os.path.exists(env_file):
            env_file = None
        super().__init__(task_schema=task.schema, task=task, env_file=env_file)
        self.enable_post_process = False  # Sentiment analysis doesn't need keyword post-processing
```

### 4. Run the Task (`run.py`)

```python
import json, os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from schema import create_sentiment_task
from config import SentimentConfig
from core.processor import RLRefineProcessor
from prompts.prompt_builder import PromptBuilder

def main():
    task = create_sentiment_task()
    config = SentimentConfig()
    builder = PromptBuilder.from_task(task)

    processor = RLRefineProcessor(config=config, task=task, prompt_builder=builder)

    texts = [
        "This phone has a large screen, but camera is mediocre",
        "Clothing quality is excellent, fabric is comfortable, fits well",
    ]

    input_data = [{"id": i, "text": text} for i, text in enumerate(texts)]
    results = processor.process_batch(input_data)

    for result in results:
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
```

Expected output:

```json
{
  "id": 0,
  "data": {
    "sentiment": "neutral",
    "confidence": 0.6,
    "aspects": [
      {"aspect": "screen", "sentiment": "positive", "confidence": 0.9},
      {"aspect": "camera", "sentiment": "negative", "confidence": 0.7}
    ]
  }
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM service URL |
| `VLLM_API_KEY` | `dummy` | vLLM API Key (usually not validated) |
| `MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | Model name/path |
| `DEBUG` | `false` | Debug mode switch |

---

## FAQ

### Q: Model output is truncated (finish_reason=length)

**Cause**: `max_tokens` is set too low, or the model is stuck in a repetitive token loop.

**Solution**:
1. Increase `config.max_token` (default 1024, try 2048)
2. Ensure `enable_thinking=False` (Thinking mode is only for RL-trained models. For untrained base models like Qwen2.5-7B-Instruct, enabling this will not improve results and will consume excessive tokens while reducing JSON output reliability)
3. The framework has built-in `frequency_penalty` + `repetition_penalty` retry mechanisms — usually 2-3 retries resolve the issue

### Q: How to use a fine-tuned model?

Simply change `MODEL_NAME` in `.env` to your fine-tuned model path, then start that model with vLLM. No framework code changes needed.

### Q: Can I use a backend other than vLLM?

In theory, yes. RLRefine uses the OpenAI-compatible API for inference, so any OpenAI API-compatible service could serve as a backend. However, please note:

**This project has only been thoroughly tested with vLLM + Qwen2.5-7B-Instruct.** When using other backends, the following parameters may not be supported or may behave differently:
- `extra_body={"repetition_penalty": ...}` — This is a vLLM-specific parameter, not supported by OpenAI's official API
- `response_format={"type": "json_object"}` — Behavior may differ across backends
- `seed` — Different backends may handle seeding differently

If you use other backends (OpenAI API, Azure OpenAI, Ollama, etc.), you may need to modify the `_build_request_params` method in `core/processor.py` to adapt.

### Q: What is `enable_thinking` and when should I use it?

`enable_thinking` controls whether the model outputs a reasoning process in `<think>...</think>` tags before the JSON result.

**Important**: This feature is designed for models that have been through the RL training pipeline (SFT → DPO → GRPO), where the model has learned structured reasoning within these tags. For untrained base models (like Qwen2.5-7B-Instruct), this parameter should always be `False` because:
- Small models don't have native "thinking" capabilities
- It conflicts with `response_format=json_object` (mutually exclusive)
- It consumes excessive tokens without improving extraction quality

### Q: How to disable post-processing?

```python
config.enable_post_process = False
```

### Q: Can I use only inference without RL training?

Absolutely. The training scripts in `rl/` are optional. You only need `core/`, `prompts/`, and `examples/` for inference tasks. You can also skip installing `ms-swift` and `vllm` when installing dependencies.

### Q: What's the difference between using and not using RL training?

Without RL training, the framework uses the base model (e.g., Qwen2.5-7B-Instruct) for inference, relying on prompt engineering and post-processing for quality. With RL training, the model itself gains better reasoning and extraction capabilities, significantly improving output quality and greatly reducing dependency on post-processing.

---

## License

Apache 2.0
