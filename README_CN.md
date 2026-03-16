[English](README.md) | [中文](README_CN.md)

# RLRefine

**用强化学习精炼 LLM 的思考过程与信息提取能力**

> Refine LLM's Reasoning & Extraction via RL (SFT → DPO → GRPO)

---

## 这个项目解决什么问题？

在用 LLM 做关键词提取等信息抽取任务时，原始模型存在以下核心问题：

- **思考过程粗糙**：模型缺乏系统性的分析步骤，经常跳过关键推理环节，导致提取结果质量不稳定
- **关键词质量差**：遗漏重要关键词、提取的词粒度不一致、无法正确处理否定词场景
- **幻觉严重**：输出的关键词不在原文中，模型自行编造或归纳总结
- **输出格式不可控**：JSON 格式错误、字段缺失、陷入重复 token 循环等

**RLRefine 的核心思路**：通过构造包含高质量思考过程的训练数据，使用 `SFT → DPO → GRPO` 三阶段 RL 训练流水线，让模型学会更规范的推理方式，从而**提升提取结果的准确率和一致性**。JSON 输出的稳定性则是训练过程中附带获得的工程收益。

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **RL 训练全流程** | 内置 SFT → DPO → GRPO 训练脚本，逐阶段精炼模型的思考与提取能力（基于 ms-swift） |
| **Schema 驱动的 Reward** | GRPO 奖励函数以**提取准确率（F1）为主导**（权重 50%），同时兼顾思考质量、格式正确性和幻觉惩罚 |
| **Schema 驱动** | 用 Python 代码定义 JSON Schema，框架自动生成 Prompt、验证响应、构建 Reward |
| **动态 Prompt 生成** | 根据 Schema 和任务规则自动生成 System/User Prompt，支持长文本/短文本自动切换 |
| **多级容错推理** | LLM 调用 → 带惩罚参数重试 → Jieba 规则兜底 → 硬截断兜底，确保生产环境永远有返回值 |
| **完整后处理流水线** | 去重、Top-N 截取、停用词过滤、时间/日期词过滤、原文对齐验证、智能回填 |
| **开箱即用的示例** | 附带关键词提取完整示例，可直接运行 |

---

## 架构概览

```
┌──────────────────────────────────────────────────────────────┐
│                      RLRefine 框架                            │
├────────────────────────────┬─────────────────────────────────┤
│                            │                                  │
│   RL 训练流水线 (核心)       │   推理引擎 (工程配套)             │
│                            │                                  │
│  ┌─────┐  ┌─────┐  ┌─────┐│  ┌──────────┐  ┌────────────┐  │
│  │ SFT │─▶│ DPO │─▶│GRPO ││  │  Schema   │─▶│  Prompt    │  │
│  └─────┘  └─────┘  └──┬──┘│  │  定义任务  │  │  Builder   │  │
│                        │   │  └──────────┘  └──────┬─────┘  │
│  ┌─────────────────────▼─┐ │                       │         │
│  │  Schema-Based Reward  │ │  ┌────────────────────▼──────┐  │
│  │                       │ │  │     LLM (vLLM) 推理       │  │
│  │  准确率 F1    (50%)   │ │  └────────────────────┬──────┘  │
│  │  Schema验证   (20%)   │ │                       │         │
│  │  格式检查     (20%)   │ │         ┌─────────────▼───────┐ │
│  │  思考质量     (10%)   │ │         │  _parse_response()  │ │
│  │  幻觉惩罚  (-0.1/词)  │ │         │  Schema 验证 JSON   │ │
│  └───────────────────────┘ │         └─────────┬───────────┘ │
│                            │                   │              │
│                            │      ┌────────────▼──────────┐  │
│                            │      │   post_process()       │  │
│                            │      │   去重/过滤/Top-N/验证  │  │
│                            │      └────────────┬──────────┘  │
│                            │                   │ 失败时       │
│                            │      ┌────────────▼──────────┐  │
│                            │      │   Jieba 兜底提取       │  │
│                            │      └───────────────────────┘  │
└────────────────────────────┴─────────────────────────────────┘
```

---

## 目录结构

```
RLRefine/
├── rl/                            # 强化学习训练模块（核心）
│   ├── reward_builder.py          #   Schema 驱动的奖励函数（用于 GRPO）
│   ├── convert_sft_to_grpo.py     #   SFT 数据格式转 GRPO 格式的工具
│   ├── sft_finetune.sh            #   SFT 训练脚本（基于 ms-swift）
│   ├── dpo_finetune.sh            #   DPO 训练脚本（基于 ms-swift）
│   └── grpo_finetune.sh           #   GRPO 训练脚本（基于 ms-swift + vLLM）
│
├── core/                          # 核心模块
│   ├── schema.py                  #   Schema 定义与验证（TaskSchema, FieldDefinition, ExtractionTask）
│   ├── config.py                  #   配置系统（支持 .env / YAML / 字典）
│   ├── processor.py               #   核心处理器（推理 + 重试 + 兜底 + 后处理）
│   ├── preprocess.py              #   文本预处理（去噪、去 Emoji、去 URL 等）
│   ├── post_process.py            #   关键词后处理（去重、过滤、Top-N、原文验证）
│   └── fallback.py                #   Jieba 兜底提取（TF-IDF / TextRank / 简单分词）
│
├── prompts/                       # Prompt 模块
│   ├── prompt_builder.py          #   动态 Prompt 生成器（根据 Schema 自动构建）
│   ├── prompt_template_3.py       #   关键词提取 Prompt 模板（长文本版）
│   └── prompt_template_4_short.py #   关键词提取 Prompt 模板（短文本版）
│
├── scripts/                       # 工具脚本
│   └── run_vllm.sh                #   一键启动 vLLM 推理服务
│
├── examples/                      # 使用示例
│   └── keyword_extraction/        #   关键词提取示例
│       ├── schema.py              #     任务 Schema 定义
│       ├── config.py              #     任务配置
│       ├── run.py                 #     运行入口
│       ├── sample_data.jsonl      #     示例数据
│       └── .env.example           #     环境变量模板
│
├── .env.example                   # 环境变量模板
├── requirements.txt               # Python 依赖
└── README.md                      # 本文件
```

---

## 安装

### 环境要求

- Python >= 3.10
- NVIDIA GPU（推理和训练均需要）
- CUDA >= 12.0

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/your-username/RLRefine.git
cd RLRefine

# 2. 安装依赖
pip install -r requirements.txt
pip install python-dotenv

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，修改 VLLM_BASE_URL 和 MODEL_NAME
```

`requirements.txt` 包含以下依赖：

```
openai>=1.0.0        # OpenAI 兼容客户端（用于调用 vLLM）
tqdm>=4.66.0         # 进度条
jieba>=0.42.1        # 中文分词（兜底提取用）
pyyaml>=6.0.0        # YAML 配置支持
ms-swift>=3.11.0     # RL 训练框架（仅训练时需要）
vllm>=0.6.0          # 推理后端（仅服务端需要）
```

> 如果你只需要推理（不需要训练），可以跳过 `ms-swift` 和 `vllm` 的安装。推理只需要 `openai`、`tqdm`、`jieba`。

---

## RL 训练流程

这是本项目的核心贡献。通过三阶段渐进式训练，逐步精炼模型的思考过程和提取能力。

### 训练流程概览

```
基座模型 (Qwen2.5-7B-Instruct)
    │
    ▼
 SFT 训练 ──────── 注入高质量思考过程的示范数据，让模型学会"怎么想"
    │
    ▼
 DPO 训练 ──────── 通过好/差对比，让模型学会区分"好的思考"和"差的思考"
    │
    ▼
 GRPO 训练 ─────── 以提取准确率为主导的 Reward 直接优化模型行为
    │
    ▼
 精炼后的模型 ──── 思考更系统、提取更准确、输出更稳定
```

### 为什么要用 RL？

原始 LLM 在关键词提取时的典型问题：

1. **思考不充分**：直接跳到结论，缺乏对文本的逐步分析
2. **遗漏关键词**：未能识别否定词场景（如"没有声音"中的"声音"）
3. **粒度不一致**：有时输出"屏幕很大"，有时输出"屏幕"，缺乏统一标准
4. **幻觉**：输出原文中不存在的词

通过 RL 训练，模型学会了：
- 系统性地拆解文本，先识别主体词再识别描述词
- 严格的原文对齐意识，不编造不存在的词
- 一致的关键词粒度（原子级关键词，≤4 个汉字）
- 规范的输出格式（JSON 稳定性作为附带收益）

### 步骤 1：准备 SFT 数据

SFT 数据是整个训练的基石。你需要构造包含**高质量思考过程**的示范数据，格式为 JSONL：

```json
{
  "messages": [
    {"role": "system", "content": "你是关键词提取专家..."},
    {"role": "user", "content": "请分析以下评论：这款手机屏幕很大..."},
    {"role": "assistant", "content": "<think>分析评论内容：1. '屏幕很大'→主体词'屏幕'，描述词'大'；2. '电池耐用'→主体词'电池'...</think>\n{\"keywords\": [[\"识别主体词'屏幕'\", \"屏幕\", 0.95], [\"识别描述词'大'\", \"大\", 0.90]]}"}
  ]
}
```

> 关键：assistant 的回复中需要包含结构化的思考过程（`<think>...</think>` 标签），这是模型需要学习的核心内容。

### 步骤 2：SFT 训练

使用 ms-swift 进行有监督微调，让模型初步学会思考和提取的模式：

```bash
# 修改 rl/sft_finetune.sh 中的路径，然后运行
bash rl/sft_finetune.sh
```

`sft_finetune.sh` 中的关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | Qwen2.5-7B-Instruct | 基座模型路径 |
| `DATASET_PATH` | `sft_data.jsonl` | SFT 训练数据路径 |
| `LORA_RANK` | `16` | LoRA 秩 |
| `LORA_ALPHA` | `64` | LoRA Alpha |
| `LEARNING_RATE` | `1e-5` | 学习率 |
| `MAX_LENGTH` | `8192` | 最大序列长度 |

### 步骤 3：DPO 训练

DPO 需要偏好数据对（chosen / rejected）。可以用更强的模型（如 Qwen-Max）自动生成。

```bash
# 修改 rl/dpo_finetune.sh 中的路径，然后运行
bash rl/dpo_finetune.sh
```

`dpo_finetune.sh` 中的关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `./output/sft_merged_model` | SFT 合并后的模型路径 |
| `DATASET_PATH` | `./data/dpo_dataset.jsonl` | DPO 训练数据路径 |
| `BETA` | `0.2` | KL 散度惩罚系数（控制偏离程度） |
| `LEARNING_RATE` | `5e-7` | 学习率（DPO 需要很小的学习率） |
| `LORA_RANK` | `8` | LoRA 秩 |

### 步骤 4：GRPO 训练

GRPO 是最关键的一步。它通过 **采样多组输出 → 计算奖励 → 策略优化** 的方式，直接以提取准确率为目标优化模型。

```bash
# 1. 将 SFT 数据转为 GRPO 格式（移除 assistant 回复，只保留 prompt）
python rl/convert_sft_to_grpo.py

# 2. 修改 rl/grpo_finetune.sh 中的路径，然后运行
bash rl/grpo_finetune.sh
```

`grpo_finetune.sh` 中的关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `./output/dpo/checkpoint-xxx-merged` | DPO 合并后的模型路径 |
| `BETA` | `0.01` | KL 惩罚系数（GRPO 需要更小的值） |
| `NUM_GENERATIONS` | `8` | 每个 prompt 采样的输出数量 |
| `TEMPERATURE` | `0.9` | 采样温度（需要足够高以产生多样性） |
| `REWARD_FUNCS` | `schema_based_reward` | 奖励函数名称 |
| `EXTERNAL_PLUGINS` | `reward_builder.py` | 奖励函数文件 |

### Reward Function（奖励函数）

`rl/reward_builder.py` 中的 `SchemaBasedReward` 对模型输出进行多维评估，**以提取准确率为绝对主导**：

| 评估维度 | 权重 | 说明 |
|----------|------|------|
| **准确率** | **0.5** | **与参考答案的 F1 Score（核心指标）** |
| **Schema 验证** | 0.2 | 必需字段是否存在、关键词项格式是否正确 |
| **格式检查** | 0.2 | JSON 是否合法、标签是否完整 |
| **思考质量** | 0.1 | 思考过程长度是否合理、是否包含分析性词汇 |
| **幻觉惩罚** | -0.1/词 | 输出的关键词不在原文中则扣分（最多扣 0.3） |

> 设计理念：准确率占 50%，因为**提取质量才是最终目标**。格式和思考过程的约束是为准确率服务的手段，而非目的本身。

---

## 快速开始

### 第一步：启动 vLLM 推理服务

RLRefine 使用 vLLM 作为推理后端（通过 OpenAI 兼容 API）。

```bash
# 方式一：直接启动
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# 方式二：使用项目提供的脚本（支持 GPU 和端口参数）
bash scripts/run_vllm.sh 0 8000    # GPU 0, 端口 8000
```

等待看到 `Application startup complete` 后，服务即就绪。

### 第二步：配置环境变量

编辑 `.env` 文件（或 `examples/keyword_extraction/.env`）：

```bash
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=dummy
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```

### 第三步：运行示例

```bash
cd RLRefine
python examples/keyword_extraction/run.py
```

你将看到类似的输出：

```json
{
  "id": 0,
  "data": {
    "keywords": ["屏幕", "电池", "拍照", "夜景"]
  }
}
```

---

## 核心概念

### 1. Schema — 定义你的 JSON 结构

`TaskSchema` 是整个框架的起点。你用它定义期望 LLM 输出的 JSON 结构，框架会自动：
- 将 Schema 嵌入 Prompt，告诉模型应该输出什么格式
- 用 Schema 的必需字段验证 LLM 响应是否合法
- 用 Schema 构建 GRPO 的 Reward 函数

```python
from core.schema import TaskSchema, FieldDefinition, FieldType, ExtractionTask

# 定义 Schema
schema = TaskSchema(
    name="keyword_extraction",
    description="电商评论关键词提取",
)

# 添加字段
schema.add_field(FieldDefinition(
    name="keywords",
    type=FieldType.ARRAY,
    description="提取的关键词列表",
    required=True,              # 必需字段：LLM 响应中必须包含
    min_length=1,               # 至少提取 1 个关键词
    max_length=15               # 最多提取 15 个关键词
))

schema.add_field(FieldDefinition(
    name="category",
    type=FieldType.STRING,
    description="评论类别",
    required=False              # 可选字段
))

# 创建任务
task = ExtractionTask(
    schema=schema,
    task_type="extraction",
    language="zh",
    domain="ecommerce",
    enable_thinking=False,      # 是否让模型先输出思考过程（仅适用于经过 RL 训练的模型，未训练模型请保持 False）
    custom_rules=[              # 任务特定规则（会嵌入 Prompt）
        "关键词长度限制在1-4个汉字",
        "必须来自原文，不可编造"
    ]
)
```

### 2. Config — 控制框架行为

`Config` 类管理所有可配置项。你可以从 `.env` 文件、YAML 文件或字典中加载配置。

```python
from core.config import Config

config = Config(task_schema=schema, task=task)

# 推理参数
config.max_token = 1024         # 最大生成 token 数
config.temperature = 0.0        # 温度（0 = 确定性输出）
config.max_retries = 3          # 最大重试次数
config.seed = 42                # 随机种子

# 惩罚参数（重试 2 次后自动启用，打破重复循环）
config.frequency_penalty = 0.3
config.repetition_penalty = 1.1

# 后处理开关
config.enable_post_process = True
config.post_process_top_n = 8                    # 保留前 N 个关键词
config.post_process_filter_stopwords = True      # 过滤停用词
config.post_process_filter_time = True           # 过滤时间词（如"8点"）
config.post_process_filter_date = True           # 过滤日期词（如"27号"）
config.post_process_filter_long = True           # 过滤超长关键词
config.post_process_max_keyword_length = 6       # 关键词最大长度
config.post_process_filter_not_in_original = True # 过滤原文中不存在的关键词
```

### 3. PromptBuilder — 自动生成 Prompt

`PromptBuilder` 根据 Schema 和任务规则自动构建 System Prompt 和 User Prompt。

有三种使用方式，你可以根据需求灵活切换（只需注释/取消注释对应代码即可）：

```python
from prompts.prompt_builder import PromptBuilder

# --- 方式一：使用内置的关键词提取 Prompt（经过实战验证的模板）---
# 适合直接用于关键词提取任务，无需手写 Prompt
prompt_builder = PromptBuilder.create_keyword_extraction_builder()

# --- 方式二：从 Task 自动生成（Schema 驱动）---
# 框架根据你定义的 Schema 和 custom_rules 自动生成 Prompt
# prompt_builder = PromptBuilder.from_task(task)

# --- 方式三：完全自定义 ---
# 当你需要完全控制 Prompt 内容时使用
# def my_prompt_generator(input_text: str, task) -> tuple:
#     system = "你是提取专家..."
#     user = f"提取：{input_text}"
#     return system, user
# prompt_builder = PromptBuilder(custom_prompt_generator=my_prompt_generator)
```

> 💡 三种方式的完整用法可参考 [`examples/keyword_extraction/run.py`](examples/keyword_extraction/run.py)。

### 4. RLRefineProcessor — 核心处理器

将上述组件组合在一起，执行完整的推理流程：

```python
from core.processor import RLRefineProcessor

processor = RLRefineProcessor(
    config=config,
    task=task,
    prompt_builder=builder
)

# 单条处理
result = processor.process_single(
    text="这款手机屏幕很大，电池也很耐用，但是拍照效果一般",
    text_id="001"
)
print(result)
# {'id': '001', 'data': {'keywords': ['屏幕', '电池', '拍照']}}

# 批量处理（自动多线程并发）
input_data = [
    {"id": "001", "describe": "这款手机屏幕很大"},
    {"id": "002", "describe": "衣服质量很好，面料舒服"},
    {"id": "003", "describe": "物流很快，包装完好"},
]
results = processor.process_batch(input_data)
```

**`process_single` 的完整执行流程：**

```
输入文本
  │
  ▼
预处理（去噪、去 Emoji、去 URL、截断）
  │
  ▼
生成 Prompt（PromptBuilder 根据 Schema 构建）
  │
  ▼
调用 LLM（通过 vLLM OpenAI 兼容 API）
  │
  ├─ 成功 ──▶ 解析 JSON ──▶ Schema 验证 ──▶ 后处理 ──▶ 返回结果
  │
  ├─ 失败 ──▶ 重试（前 2 次用常规参数，后 2 次加惩罚参数）
  │
  └─ 全部失败 ──▶ Jieba 兜底提取 ──▶ 返回兜底结果
```

---

## 推理引擎的容错机制

在生产环境中，LLM 推理不可能 100% 成功，必须有完善的容错机制。这是本框架的工程配套能力。

### 第一层：带惩罚参数重试

当 LLM 输出 JSON 解析失败时，框架会自动重试。前 2 次使用常规参数，从第 3 次开始加入 `frequency_penalty` 和 `repetition_penalty`，打破模型可能陷入的重复 token 循环。

```python
# processor.py 中的核心逻辑
while retry_count <= max_retries and parsed_data is None:
    use_penalty = retry_count >= 2  # 第 3 次开始加惩罚
    response = self._call_llm(system_prompt, user_prompt, use_penalty, retry_count)
    parsed_data = self._parse_response(response)
    retry_count += 1
```

### 第二层：Jieba 兜底提取

当 LLM 完全失败时，降级为基于规则的 Jieba 提取：

- **TF-IDF** (默认)：基于词频-逆文档频率提取关键词
- **TextRank**：基于图算法提取关键词
- **简单分词**：按词性（名词、形容词）提取

### 第三层：响应解析容错

`_parse_response` 方法不依赖 LLM 返回完美的纯 JSON，而是：

1. 使用正则 `re.search(r'\{.*\}', response)` 从任意文本中提取 JSON 片段
2. 如果启用了 Thinking 模式，先剥离 `<tag>...</tag>` 标签
3. 基于 Schema 的必需字段验证，而非硬编码检查固定字段名

---

## 自定义任务

以下是创建一个全新任务（如情感分析）的完整步骤。

### 1. 创建任务目录

```
examples/
└── sentiment_analysis/
    ├── schema.py
    ├── config.py
    ├── run.py
    └── .env
```

### 2. 定义 Schema (`schema.py`)

```python
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from core.schema import TaskSchema, FieldDefinition, FieldType, ExtractionTask

def create_sentiment_task() -> ExtractionTask:
    schema = TaskSchema(name="sentiment", description="文本情感分析")

    schema.add_field(FieldDefinition(
        name="sentiment",
        type=FieldType.STRING,
        description="情感倾向",
        required=True,
        enum_values=["positive", "negative", "neutral"]
    ))

    schema.add_field(FieldDefinition(
        name="confidence",
        type=FieldType.FLOAT,
        description="置信度",
        required=True,
        min_value=0.0,
        max_value=1.0
    ))

    schema.add_field(FieldDefinition(
        name="aspects",
        type=FieldType.ARRAY_OF_OBJECTS,
        description="方面级情感",
        required=False
    ))

    return ExtractionTask(
        schema=schema,
        task_type="extraction",
        language="zh",
        domain="ecommerce",
        enable_thinking=False,
        custom_rules=[
            "综合分析正面和负面情感",
            "置信度反映情感明确程度"
        ]
    )
```

### 3. 定义配置 (`config.py`)

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
        self.enable_post_process = False  # 情感分析不需要关键词后处理
```

### 4. 运行任务 (`run.py`)

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
        "这款手机屏幕很大，但是拍照效果一般",
        "衣服质量非常好，面料舒服，穿起来合身",
    ]

    input_data = [{"id": i, "text": text} for i, text in enumerate(texts)]
    results = processor.process_batch(input_data)

    for result in results:
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
```

运行后，你会得到类似的输出：

```json
{
  "id": 0,
  "data": {
    "sentiment": "neutral",
    "confidence": 0.6,
    "aspects": [
      {"aspect": "屏幕", "sentiment": "positive", "confidence": 0.9},
      {"aspect": "拍照", "sentiment": "negative", "confidence": 0.7}
    ]
  }
}
```

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM 服务地址 |
| `VLLM_API_KEY` | `dummy` | vLLM API Key（通常不检查） |
| `MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | 模型名称/路径 |
| `DEBUG` | `false` | 调试模式开关 |

---

## 常见问题

### Q: 模型输出被截断（finish_reason=length）

**原因**：`max_tokens` 设置过小，或模型陷入重复 token 循环。

**解决方案**：
1. 增大 `config.max_token`（默认 1024，建议试试 2048）
2. 确保 `enable_thinking=False`（Thinking 模式仅适用于经过 RL 训练的模型。对于未经训练的原始模型（如 Qwen2.5-7B-Instruct），开启此选项不仅不会提升效果，还会消耗大量 token 并降低 JSON 输出的可靠性）
3. 框架已内置 `frequency_penalty` + `repetition_penalty` 重试机制，通常 2-3 次重试即可解决

### Q: 如何使用微调后的模型？

只需修改 `.env` 中的 `MODEL_NAME` 为你微调后的模型路径，然后用 vLLM 启动该模型即可。框架代码无需任何修改。

### Q: 可以不使用 vLLM 吗？

理论上可以。RLRefine 使用 OpenAI 兼容 API 进行推理，理论上任何兼容 OpenAI API 的服务都可以作为后端。但请注意：

**本项目仅在 vLLM + Qwen2.5-7B-Instruct 上经过充分测试。** 使用其他后端时，以下参数可能不被支持或行为不同：
- `extra_body={"repetition_penalty": ...}` — 这是 vLLM 特有的参数，OpenAI 官方 API 不支持
- `response_format={"type": "json_object"}` — 不同服务端的行为可能不同
- `seed` — 不同服务端对种子的处理方式可能不同

如果你使用其他后端（如 OpenAI API、Azure OpenAI、Ollama 等），可能需要修改 `core/processor.py` 中的 `_build_request_params` 方法来适配

### Q: 如何禁用后处理？

```python
config.enable_post_process = False
```

### Q: 如何只使用推理功能，不做 RL 训练？

完全可以。`rl/` 目录下的训练脚本是可选的。你只需要 `core/`、`prompts/`、`examples/` 即可完成推理任务。安装依赖时也可以跳过 `ms-swift` 和 `vllm`。

### Q: 不做 RL 训练和做了 RL 训练的区别？

不做 RL 训练时，框架使用原始基座模型（如 Qwen2.5-7B-Instruct）进行推理，依赖 Prompt 工程和后处理来保证质量。做了 RL 训练后，模型本身具备了更好的思考和提取能力，输出质量显著提升，对后处理的依赖也大幅降低。

---

## License

Apache 2.0
