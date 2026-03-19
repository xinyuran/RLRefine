English | [中文](README.md)

# RLRefine

**Schema-driven structured information extraction framework with RL training enhancement**

> Define a Schema to drive any structured extraction task with LLMs. Need higher quality? Use RL training (SFT → DPO → GRPO) to refine the model's reasoning and extraction capabilities.

> **Note**: This project was primarily designed and tested for **Chinese e-commerce review keyword extraction**. While the framework architecture is language-agnostic, the built-in prompts and post-processing are optimized for Chinese text. Users should test and adapt for their own use cases.

---

## Results

Keyword extraction on a Chinese e-commerce review, comparing base Qwen2.5-7B-Instruct vs. RL-trained model:

```text
终于收到我需要的宝贝了，东西很好，价美物廉，谢谢掌柜的！说实在，这是我购物以来来让我最满意的一次购物。
无论是掌柜的态度还是对物品，我都非常满意的。掌柜态度很专业热情，有问必答，回复也很快……再说宝贝，
正是我需要的，收到的时候包装完整，打开后让我惊喜的是，宝贝比我想象中的还要好！
```

| Dimension | Base Model | RL-Trained Model |
|-----------|-----------|-----------------|
| Keywords extracted | 6 | 8 |
| Reasoning structure | Shallow Markdown list | Systematic 5-step analysis |
| Confidence type | String `"0.95"` (wrong) | Number `0.95` (correct) |
| Missed keywords | "满意", "好", "包装" | None |

**Base model**: `['宝贝', '态度', '回复', '价美物廉', '购物', '优惠']`

**RL-trained**: `['满意', '好', '宝贝', '价美物廉', '态度', '回复', '包装', '优惠']`

<details>
<summary><b>Click to expand: Full model output comparison</b></summary>

#### Base Qwen2.5-7B-Instruct output (626 chars, ~8 seconds)

```text
### 思考
1. **主体定位**：
   - 主体对象：宝贝、掌柜、态度、物品、包装。
2. **评价、描述定位**：
   - "东西很好"：主体"东西"，描述"好"。
   - "价美物廉"：主体"价美物廉"。
   - "这是我购物以来让我最满意的一次购物"：主体"购物"，描述"满意"。
   - "掌柜态度很专业热情"：主体"态度"，描述"专业热情"。
   - "有问必答，回复也很快"：主体"回复"，描述"快"。
   - "宝贝比我想象中的还要好"：主体"宝贝"，描述"好"。
3. **补充名词**：
   - "优惠"：与购物相关的重要名词。
4. **否定与问题场景**：
   - 无明显否定结构。

### JSON输出
{
  "keywords": [
    ["宝贝收到后包装完整", "宝贝", "0.95"],
    ["宝贝比我想象中的还要好", "宝贝", "0.95"],
    ["掌柜态度很专业热情", "态度", "0.90"],
    ["有问必答，回复也很快", "回复", "0.85"],
    ["价美物廉", "价美物廉", "0.80"],
    ["这是我购物以来让我最满意的一次购物", "购物", "0.75"],
    ["掌柜给我个优惠哦", "优惠", "0.70"],
    ["东西很好", "东西", "0.65"]
  ]
}
```

Issues: Shallow reasoning (simple list), confidence scores are strings `"0.95"` instead of numbers, missed "满意", "好", "包装".

#### RL-trained model output (1,691 chars, ~23 seconds)

```text
思考
首先通读整条评论，识别主体对象和评价描述。评论围绕一次购物体验展开，核心主体包括
"宝贝"、"包装"、"掌柜的态度"、"回复速度"等。根据规则，需提取原子级关键词，
且每个关键词≤4字，忠于原文，不可改写。

第一步：提取全局主体对象。文中多次提到"宝贝"，是核心商品主体；"包装"是物流/交付
环节的关键属性；"掌柜"指代卖家或客服角色，属于服务主体；"回复"关联到客服响应效率。

第二步：逐句分析评价与描述：
- "东西很好" → "好"可作为通用评价词保留。
- "价美物廉"是固定四字成语，符合规则，整体作为原子关键词。
- "最满意的一次购物" → "满意"是核心情绪词，需提取。
- "掌柜的态度很专业热情" → 拆出"态度"、"专业"、"热情"。
- "有问必答，回复也很快" → "回复"是服务行为，"快"是描述。
- "包装完整" → "包装"是对象，"完整"是状态描述。
- "给个优惠" → "优惠"是价格相关诉求名词。

第三步：检查否定/问题场景——本评论无负面内容，无需处理否定结构。

第四步：过滤冗余与合并——"东西" vs "宝贝"优先保留"宝贝"。

第五步：控制数量≤15，并按重要性排序。

{
  "keywords": [
    ["评论多次强调整体购物体验满意，'满意'为核心情绪词", "满意", 0.95],
    ["'东西很好'中提取通用正面评价词'好'", "好", 0.90],
    ["全文核心商品主体，多次提及'宝贝'", "宝贝", 0.88],
    ["固定四字成语'价美物廉'直接引用", "价美物廉", 0.85],
    ["'掌柜的态度'中提取服务对象词'态度'", "态度", 0.82],
    ["'回复也很快'中提取服务行为关键词'回复'", "回复", 0.80],
    ["'包装完整'中提取交付属性关键词'包装'", "包装", 0.78],
    ["'给个优惠'中提取价格相关诉求名词'优惠'", "优惠", 0.75]
  ]
}
```

Improvements: Systematic 5-step reasoning, correct numeric confidence `0.95`, each keyword has an extraction rationale, no missed keywords.

</details>

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

> **Inference only** requires `openai`, `tqdm`, `jieba` — skip `ms-swift` and `vllm`.
> **RL training** additionally requires `ms-swift` and `vllm`.

---

## Architecture Overview

RLRefine consists of two independent modules:

| | Module 1: Schema-Driven Inference | Module 2: RL Training Enhancement |
|---|---|---|
| **Purpose** | Define Schema, use LLM for structured extraction | Improve model reasoning and extraction via RL |
| **Cost** | Low — define Schema + deploy vLLM | High — requires GPU training environment |
| **Dependencies** | `openai`, `tqdm`, `jieba` | `ms-swift`, `vllm` |
| **Required** | Yes | No (optional enhancement) |

**Schema is the core of the framework** — defined once, shared by inference and training:

```
┌───────────────────────────────────────────────┐
│                    Schema                     │
│    (Define task structure in Python, reused)  │
└──────────────────┬────────────────────────────┘
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
┌──────────────┐      ┌──────────────────┐
│ Module 1:    │      │ Module 2:        │
│ Inference    │      │ RL Training      │
│              │      │                  │
│ PromptBuilder│      │  Reward Builder  │
│ → LLM Call   │      │  SFT → DPO → GRPO│
│ → Validation │      │  → Refined Model │
│ → Post-proc  │      │                  │
└──────────────┘      └──────────────────┘
```

---

## Module 1: Schema-Driven Inference

> Low cost: Define Schema → Generate Prompt → Call LLM → Structured output

### Quick Start

**1. Start vLLM service**

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# Or use the project script
bash scripts/run_vllm.sh 0 8000    # GPU 0, Port 8000
```

**2. Configure environment variables**

```bash
cp .env.example .env
# Edit .env:
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=dummy
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```

**3. Run the example**

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

### Core API

#### 1. Schema — Define Task Structure

`TaskSchema` is the core of the framework, defining field structures and constraints for extraction tasks:

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

Supported field types: `STRING`, `INTEGER`, `FLOAT`, `BOOLEAN`, `ARRAY`, `OBJECT`, `ARRAY_OF_OBJECTS`

Each field supports constraints: `required`, `min_value`/`max_value`, `enum_values`, `min_length`/`max_length`, `pattern`, etc.

#### 2. PromptBuilder — Generate Prompts

Three methods, from out-of-the-box to fully custom:

```python
from prompts.prompt_builder import PromptBuilder

# Method 1: Built-in template (keyword extraction, battle-tested)
prompt_builder = PromptBuilder.create_keyword_extraction_builder()

# Method 2: Auto-generate from Schema (for custom tasks)
prompt_builder = PromptBuilder.from_task(task)

# Method 3: Fully custom prompt generator
def my_generator(input_text: str, task) -> tuple:
    return "You are an extraction expert...", f"Extract: {input_text}"
prompt_builder = PromptBuilder(custom_prompt_generator=my_generator)
```

> **Method 1** uses `prompt_template_3.py`, instructing the model to output reasoning before JSON. `config.response_format` must remain `None`.
> **Method 2** auto-generates prompts from Schema. Set `config.response_format = {"type": "json_object"}` for pure JSON output.

#### 3. Config — Tune Behavior

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
```

#### 4. Processor — Run Extraction

```python
from core.processor import RLRefineProcessor

processor = RLRefineProcessor(config=config, task=task, prompt_builder=prompt_builder)

# Single item
result = processor.process_single(text="...", text_id="001")

# Batch (10 threads by default)
results = processor.process_batch([
    {"id": "001", "describe": "..."},
    {"id": "002", "describe": "..."},
])
```

**Execution flow:**
```
Input → Prompt → LLM Call → Parse JSON → Schema Validate → Post-process → Output
                                ↓ on failure
                          Retry (+ penalty params after 2 fails)
                                ↓ all failed
                          Jieba TF-IDF fallback → Output
```

### Custom Task Example: Sentiment Analysis

Define a new Schema to use the framework for an entirely different extraction task:

```python
from core.schema import TaskSchema, FieldDefinition, FieldType, ExtractionTask
from core.config import Config
from core.processor import RLRefineProcessor
from prompts.prompt_builder import PromptBuilder

# 1. Define Schema
schema = TaskSchema(name="sentiment", description="Text sentiment analysis")
schema.add_field(FieldDefinition(
    name="sentiment", type=FieldType.STRING, required=True,
    enum_values=["positive", "negative", "neutral"]
))
schema.add_field(FieldDefinition(
    name="confidence", type=FieldType.FLOAT, required=True,
    min_value=0.0, max_value=1.0
))
task = ExtractionTask(schema=schema, custom_rules=["Confidence reflects sentiment clarity"])

# 2. Configure (Schema-driven, enforce pure JSON output)
config = Config(task_schema=task.schema, task=task)
config.enable_post_process = False
config.response_format = {"type": "json_object"}

# 3. Build prompt and run
builder = PromptBuilder.from_task(task)
processor = RLRefineProcessor(config=config, task=task, prompt_builder=builder)
results = processor.process_batch([{"id": "1", "text": "Great product!"}])
```

---

## Module 2: RL Training Enhancement

> High cost: requires GPU training environment, `ms-swift`, `vllm`

Three-stage RL training teaches the model to reason systematically before extraction, significantly improving quality.

### Training Flow

```
Qwen2.5-7B-Instruct (base)
    │
    ▼ SFT  ── Teach reasoning patterns via demonstrations
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

> After each stage, use `rl/merge_lora.sh` to merge LoRA weights into the base model before proceeding.

### Step 1: Prepare Training Data

**SFT data**: Use a strong model (e.g., Qwen3-Max) to generate high-quality reasoning + extraction demonstrations. See `rl/generate_data_qwen3-max.py` and `prompts/prompt_template_3.py`.

```json
{
  "messages": [
    {"role": "system", "content": "(prompt from prompt_template_3.py)"},
    {"role": "user", "content": "Analyze this review: ..."},
    {"role": "assistant", "content": "**思考**\nStep 1: Identify subjects...\n\n**JSON输出**\n{\"keywords\": [...]}"}
  ]
}
```

**DPO data**: Use `rl/generate_data_dpo.py` with `prompts/prompt_template_dpo.py` to generate chosen/rejected preference pairs.

**GRPO data**: Convert SFT data to GRPO format (remove assistant responses, keep only prompts):

```bash
python rl/convert_sft_to_grpo.py
```

> For data format requirements, refer to the [ms-swift custom dataset documentation](https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-dataset.html#dpo-orpo-cpo-simpo-rm).

### Step 2: SFT Training

```bash
bash rl/sft_finetune.sh
bash rl/merge_lora.sh
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
bash rl/merge_lora.sh
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_PATH` | `./output/sft/xxx-merged` | SFT merged model |
| `BETA` | `0.2` | KL penalty coefficient |
| `LEARNING_RATE` | `5e-7` | Learning rate (very small for DPO) |

### Step 4: GRPO Training

```bash
bash rl/grpo_finetune.sh
bash rl/merge_lora.sh
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_PATH` | `./output/dpo/xxx-merged` | DPO merged model |
| `BETA` | `0.01` | KL penalty (smaller than DPO) |
| `NUM_GENERATIONS` | `8` | Outputs sampled per prompt |
| `TEMPERATURE` | `0.9` | Sampling temperature |

### Reward Design

Accuracy at 50% weight — **extraction quality is the goal**, format and reasoning constraints are means:

| Dimension | Weight | Purpose |
|-----------|--------|---------|
| Accuracy (F1) | **50%** | Core extraction quality |
| Schema Validation | 20% | Output correctness |
| Format Check | 20% | JSON reliability |
| Thinking Quality | 10% | Reasoning robustness |
| Hallucination Penalty | -0.1/word | Anti-fabrication |

---

## Directory Structure

```
RLRefine/
├── core/                              # Core inference module
│   ├── schema.py                      #   Schema definition & validation
│   ├── config.py                      #   Configuration system
│   ├── processor.py                   #   Core processor (inference + retry + fallback)
│   ├── preprocess.py                  #   Text preprocessing
│   ├── post_process.py                #   Keyword post-processing
│   └── fallback.py                    #   Jieba fallback (TF-IDF / TextRank)
│
├── prompts/                           # Prompt module
│   ├── prompt_builder.py              #   Dynamic prompt generator
│   ├── prompt_template_3.py           #   Keyword extraction template (with reasoning)
│   ├── prompt_template_4_short.py     #   Keyword extraction template (short text)
│   └── prompt_template_dpo.py         #   DPO data generation template
│
├── rl/                                # RL training module
│   ├── reward_builder.py              #   Schema-driven reward function (GRPO)
│   ├── generate_data_qwen3-max.py     #   SFT data generation (reference)
│   ├── generate_data_dpo.py           #   DPO data generation (reference)
│   ├── convert_sft_to_grpo.py         #   SFT → GRPO data format converter
│   ├── merge_lora.sh                  #   LoRA weight merging script
│   ├── sft_finetune.sh                #   SFT training script (ms-swift)
│   ├── dpo_finetune.sh                #   DPO training script (ms-swift)
│   └── grpo_finetune.sh               #   GRPO training script (ms-swift + vLLM)
│
├── scripts/                           # Utility scripts
│   └── run_vllm.sh                    #   vLLM startup script
│
├── examples/
│   └── keyword_extraction/            #   Complete runnable example
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## License

Apache 2.0
