[中文](README_CN.md) | English

# RLRefine

**RL-Enhanced Structured Information Extraction with LLMs**

> RLRefine is an end-to-end practice of applying RL training (SFT → DPO → GRPO) to LLM-based structured information extraction. Using Chinese e-commerce review keyword extraction as the core scenario, it demonstrates the complete pipeline from Schema definition, Prompt engineering, multi-stage RL training to reward function design. The framework is inherently general-purpose — simply swap the Schema definition to adapt to other extraction tasks.

---

## Model Weights

Trained model weights are available on HuggingFace: [https://huggingface.co/xinyuran/Qwen2.5-7B-RLRefine](https://huggingface.co/xinyuran/Qwen2.5-7B-RLRefine)

---

## Highlights

- **Complete RL Training Pipeline**: Three-stage progressive training (SFT → DPO → GRPO), completed on 2×H100 based on Qwen2.5-7B-Instruct
- **Schema-Driven Generality**: Define a Schema once, share it across inference and training; swap Schemas to adapt to sentiment analysis, entity extraction, etc.
- **Explainable Reward Function**: Five-dimension composite reward (F1 50% + Schema validation 20% + Format 20% + Thinking quality 10% + Hallucination penalty), with complete design documentation
- **End-to-End Reproducibility**: Training scripts, data samples, training logs, and evaluation methodology provided

---

## Results

Comparison on Chinese e-commerce keyword extraction between vanilla Qwen2.5-7B-Instruct and RL-trained model:

| Dimension | Vanilla Model | RL-Trained Model |
|-----------|--------------|-----------------|
| Keywords extracted | 6 | 8 |
| Reasoning structure | Simple Markdown list | Systematic 5-step analysis |
| Confidence type | String `"0.95"` (incorrect) | Number `0.95` (correct) |
| Missing keywords | "满意", "好", "包装" | None |

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

---

## Architecture

RLRefine consists of two independent modules:

| | Module 1: Schema-Driven Inference | Module 2: RL Training Enhancement |
|---|---|---|
| **Purpose** | Define Schema, use LLM for structured extraction | Improve model reasoning and extraction quality via RL |
| **Cost** | Low — define Schema + deploy vLLM | High — requires GPU training environment |
| **Dependencies** | `openai`, `tqdm`, `jieba`, `vllm` | `ms-swift` |
| **Required?** | Yes | No (optional enhancement) |

**Schema is the core** — define once, used by both inference and training:

```
+-----------------------------------------------+
|                    Schema                     |
|     (Python code defines task structure)      |
+------------------+----------------------------+
                   |
       +-----------+-----------+
       v                       v
+--------------+      +------------------+
| Module 1:    |      | Module 2:        |
| Inference    |      | RL Training      |
|              |      |                  |
| PromptBuilder|      | Reward Builder   |
| -> LLM Call  |      | SFT -> DPO -> GRPO|
| -> Validation|      | -> Refined Model  |
| -> Post-proc |      |                  |
+--------------+      +------------------+
```

---

## Module 1: Schema-Driven Inference

> Low cost: Define Schema → Generate Prompt → Call LLM → Structured Output

### Quick Start

**1. Start vLLM server**

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# Or use the project script
bash scripts/run_vllm.sh 0 8000    # GPU 0, port 8000
```

**2. Configure environment**

```bash
cp .env.example .env
# Edit .env:
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=dummy
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```

**3. Run example**

```bash
cd RLRefine
python examples/keyword_extraction/run.py
```

### Core API

#### 1. Schema — Define Task Structure

```python
from core.schema import TaskSchema, FieldDefinition, FieldType, ExtractionTask

schema = TaskSchema(name="keyword_extraction", description="E-commerce review keyword extraction")
schema.add_field(FieldDefinition(
    name="keywords",
    type=FieldType.ARRAY_OF_OBJECTS,
    description="List of extracted keywords",
    required=True,
))

task = ExtractionTask(
    schema=schema,
    language="zh",
    domain="ecommerce",
    custom_rules=["Keywords must come from original text, no fabrication"]
)
```

#### 2. PromptBuilder — Generate Prompts

```python
from prompts.prompt_builder import PromptBuilder

# Option 1: Built-in template (keyword extraction, battle-tested)
prompt_builder = PromptBuilder.create_keyword_extraction_builder()

# Option 2: Auto-generate from Schema (for custom tasks)
prompt_builder = PromptBuilder.from_task(task)

# Option 3: Fully custom prompt generator
def my_generator(input_text: str, task) -> tuple:
    return "You are an extraction expert...", f"Extract: {input_text}"
prompt_builder = PromptBuilder(custom_prompt_generator=my_generator)
```

#### 3. Processor — Execute Extraction

```python
from core.processor import RLRefineProcessor

processor = RLRefineProcessor(config=config, task=task, prompt_builder=prompt_builder)

# Single item
result = processor.process_single(text="...", text_id="001")

# Batch processing (10 concurrent threads by default)
results = processor.process_batch([
    {"id": "001", "describe": "..."},
    {"id": "002", "describe": "..."},
])
```

**Processing flow:**
```
Input -> Prompt -> LLM Call -> JSON Parse -> Schema Validate -> Post-process -> Output
                                  | failure
                            Retry (penalty params on 3rd attempt)
                                  | all failed
                            Jieba TF-IDF fallback -> Output
```

---

## Module 2: RL Training Enhancement

> High cost: Requires GPU training environment, `ms-swift`, `vllm`

Three-stage RL training teaches the model systematic reasoning before extraction.

### Training Pipeline

```
Qwen2.5-7B-Instruct (base)
    |
    v SFT  -- Teach reasoning patterns via demonstration
    |
    v DPO  -- Teach quality preferences via good/bad pairs
    |
    v GRPO -- Optimize directly with extraction F1 as reward
    |
    v
Refined Model (systematic reasoning + accurate extraction)
```

### Training Configuration Summary

| Stage | Learning Rate | LoRA Rank | Epochs | Duration | Peak VRAM |
|-------|--------------|-----------|--------|----------|-----------|
| SFT | 1e-5 | 16 | 5 | ~1h 9min | 77.4 GiB |
| DPO | 5e-7 | 8 | 1 | -- | -- |
| GRPO | 5e-7 | 8 | 1 | ~4h 35min | 66.7 GiB |

Hardware: 2× NVIDIA H100 80GB HBM3

> Full training details, loss curves, and engineering notes: [docs/training_details.md](docs/training_details.md)

### Reward Design

Accuracy weight at 50% — **extraction quality is the ultimate goal**:

| Dimension | Weight | Role |
|-----------|--------|------|
| Accuracy (F1) | **50%** | Core extraction quality |
| Schema validation | 20% | Output correctness |
| Format check | 20% | JSON reliability |
| Thinking quality | 10% | Reasoning robustness |
| Hallucination penalty | -0.1/word | Prevent fabrication |

> Detailed reward function design and Reward Hacking analysis: [docs/reward_design.md](docs/reward_design.md)

---

## Documentation

| Document | Content |
|----------|---------|
| [docs/reward_design.md](docs/reward_design.md) | Reward function design: five-dimension weights, scoring rules, Reward Hacking analysis |
| [docs/training_details.md](docs/training_details.md) | Training details: hardware, per-stage config, loss curves, engineering issues |
| [docs/evaluation.md](docs/evaluation.md) | Evaluation: production test results and limitations |
| [examples/sample_data/](examples/sample_data/) | Training data samples (SFT / DPO / GRPO formats) |

---

## Project Structure

```
RLRefine/
+-- core/                              # Core inference module
|   +-- schema.py                      #   Schema definition & validation
|   +-- config.py                      #   Configuration system
|   +-- processor.py                   #   Core processor (inference + retry + fallback)
|   +-- preprocess.py                  #   Text preprocessing
|   +-- post_process.py                #   Keyword post-processing
|   +-- fallback.py                    #   Jieba fallback (TF-IDF / TextRank)
+-- prompts/                           # Prompt module
|   +-- prompt_builder.py              #   Dynamic prompt generator
|   +-- prompt_template_3.py           #   Keyword extraction template (with reasoning)
+-- rl/                                # RL training module
|   +-- reward_builder.py              #   Schema-driven reward function (GRPO)
|   +-- sft_finetune.sh                #   SFT training script
|   +-- dpo_finetune.sh                #   DPO training script
|   +-- grpo_finetune.sh               #   GRPO training script (ms-swift + vLLM)
|   +-- train_logs/                    #   Training logs
+-- docs/                              # Detailed documentation
|   +-- reward_design.md               #   Reward function design
|   +-- training_details.md            #   Training details & engineering notes
|   +-- evaluation.md                  #   Evaluation methodology & results
+-- examples/
|   +-- keyword_extraction/            #   Complete runnable example
|   +-- sample_data/                   #   Training data samples
+-- .env.example
+-- requirements.txt
+-- README.md
```

---

## License

MIT
