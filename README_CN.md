[English](README.md) | [中文](README_CN.md)

# RLRefine

**用强化学习精炼 LLM 的思考过程与信息提取能力（SFT → DPO → GRPO）**

> 一个训练小型 LLM 进行结构化信息提取、并具备系统性推理能力的框架。

---

## 1. 问题

在用小型 LLM（如 Qwen2.5-7B-Instruct）做结构化提取任务时，原始基座模型存在以下典型失效模式：

| 问题 | 表现 |
|------|------|
| **思考粗糙** | 跳过分析步骤，直接输出结论 |
| **幻觉** | 输出的词不在原文中 |
| **粒度不一致** | 有时输出"屏幕很大"，有时输出"屏幕"，没有统一标准 |
| **格式不稳定** | JSON 格式错误、字段缺失、重复 token 循环 |
| **否定词处理失败** | 遇到"没有声音"时，无法提取"声音" |

这些问题让小模型难以在生产环境中可靠地完成提取任务。

---

## 2. 方案

RLRefine 通过三阶段训练流水线，教会模型在提取前**系统性地思考**：

```
基座模型  →  SFT  →  DPO  →  GRPO  →  精炼后模型
             （学习   （偏好    （优化     （系统性
              模式）   质量）   准确率）    推理）
```

核心思想：构造含高质量思考过程的训练数据，再用 RL 强化提取准确率。Schema 驱动的设计确保同一个任务定义同时用于**训练（奖励计算）和推理（Prompt 生成 + 响应验证）**。

**推理过程的工作方式**：Prompt 模板（`prompt_template_3.py`）指示模型先输出"思考"部分，再输出 JSON 结果。这个推理结构直接内嵌在 Prompt 中——模型始终接收这些指令。RL 训练后，模型产出的推理和提取质量会显著提升。

---

## 3. 效果

用同一条中文电商评论（222 字），对比 Qwen2.5-7B-Instruct 原始模型和 RL 训练后模型的真实表现：

| 维度 | 原始模型 | RL 训练后模型 |
|------|---------|--------------|
| 推理耗时 | ~8 秒 | ~23 秒 |
| 输出字符数 | 626 字 | 1,691 字 |
| 提取关键词数 | 6 个 | 8 个 |
| 推理结构 | 简单 Markdown 列表 | 系统化五步分析 |
| 置信度类型 | 字符串 `"0.95"`（错误） | 数字 `0.95`（正确） |
| 幻觉 | 无 | 无 |
| 遗漏关键词 | "满意"、"好"、"包装" | 无 |

**原始模型结果**：`['宝贝', '态度', '回复', '价美物廉', '购物', '优惠']`

**RL 训练后结果**：`['满意', '好', '宝贝', '价美物廉', '态度', '回复', '包装', '优惠']`

RL 训练后的模型正确识别了"满意"为核心情绪词，提取到了原始模型完全遗漏的"包装"，且置信度类型完全正确。

> **延迟权衡**：RL 训练后模型耗时约为原始模型的 3 倍，因为输出了更详细的推理过程（token 量约 3 倍）。这是有意义的质量-延迟权衡：模型学会了在提取前进行更充分的思考。详见 [性能说明](#性能说明)。

---

## 4. 架构

**Schema 是整个框架的核心组织原则**——定义一次，训练和推理共同使用：

```
┌─────────────────────────────────────────────────────────────┐
│                         Schema                               │
│           （用 Python 代码定义任务结构，一次定义，全局使用）    │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┴──────────────┐
         │                            │
         ▼                            ▼
┌─────────────────┐        ┌──────────────────────────┐
│   训练阶段       │        │        推理阶段             │
│                 │        │                           │
│  Reward Builder │        │  Prompt Builder           │
│  ┌────────────┐ │        │  （根据 Schema 自动生成）   │
│  │准确率 F1   │ │        │           │               │
│  │ (50%)     │ │        │           ▼               │
│  │Schema验证  │ │        │  LLM 推理 (vLLM)          │
│  │ (20%)     │ │        │           │               │
│  │格式检查    │ │        │   解析 + Schema 验证       │
│  │ (20%)     │ │        │   （必需字段 + JSON 提取）  │
│  │思考质量    │ │        │           │               │
│  │ (10%)     │ │        │   后处理                   │
│  │幻觉惩罚   │ │        │   （去重/过滤/Top-N）       │
│  │(-0.1/词)  │ │        │           │               │
│  └─────┬──────┘ │        │   输出 / Jieba 兜底        │
│        │        │        └──────────────────────────┘
│  SFT→DPO→GRPO  │
│        │        │
│  训练后模型 ────┼──────────────────────▶ （用于推理）
└─────────────────┘
```

---

## 5. 核心思想

### Schema 驱动设计

同一个 `TaskSchema` 对象驱动所有流程：
- **训练**：Reward 函数用 Schema 计算 F1、验证格式、惩罚幻觉
- **推理**：PromptBuilder 将 Schema 嵌入 Prompt；处理器用 Schema 必需字段验证响应

### 三阶段 RL 训练

每个阶段解决不同问题：
- **SFT**：通过示范数据教会*什么是好的推理过程*
- **DPO**：通过好/差对比教会*质量偏好*
- **GRPO**：以提取准确率（F1）为主要奖励信号，直接优化*提取效果*

### 奖励设计理念

准确率权重 50%——因为**提取质量才是最终目标**，格式和思考约束是实现目标的手段：

| 维度 | 权重 | 作用 |
|------|------|------|
| 准确率（F1） | **50%** | 核心提取质量 |
| Schema 验证 | 20% | 输出正确性 |
| 格式检查 | 20% | JSON 可靠性 |
| 思考质量 | 10% | 推理鲁棒性 |
| 幻觉惩罚 | -0.1/词 | 防止编造 |

### 生产级推理引擎

多层容错确保永远有返回值：
```
LLM 调用 → [带惩罚参数重试] → [Jieba TF-IDF 兜底] → 输出
```

---

## 6. 快速开始

### 第一步：启动 vLLM

```bash
# 方式一：直接启动
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# 方式二：使用项目脚本
bash scripts/run_vllm.sh 0 8000    # GPU 0, 端口 8000
```

等待看到 `Application startup complete`。

### 第二步：配置

```bash
cp .env.example .env
# 编辑 .env：
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=dummy
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```

### 第三步：运行示例

```bash
cd RLRefine
python examples/keyword_extraction/run.py
```

预期输出：

```json
{
  "id": 0,
  "data": {
    "keywords": ["屏幕", "电池", "拍照", "夜景"]
  }
}
```

---

## 安装

### 环境要求

- Python >= 3.10
- NVIDIA GPU + CUDA >= 12.0

### 安装步骤

```bash
git clone https://github.com/your-username/RLRefine.git
cd RLRefine
pip install -r requirements.txt
pip install python-dotenv
```

> 仅推理只需要 `openai`、`tqdm`、`jieba`，可跳过 `ms-swift` 和 `vllm` 的安装。

---

## 核心 API

### Schema — 定义任务

```python
from core.schema import TaskSchema, FieldDefinition, FieldType, ExtractionTask

schema = TaskSchema(name="keyword_extraction", description="电商评论关键词提取")
schema.add_field(FieldDefinition(
    name="keywords",
    type=FieldType.ARRAY_OF_OBJECTS,
    description="提取的关键词列表",
    required=True,
))

task = ExtractionTask(
    schema=schema,
    language="zh",
    domain="ecommerce",
    custom_rules=["关键词必须来自原文，不可编造"]
)
```

### PromptBuilder — 三种方式

```python
from prompts.prompt_builder import PromptBuilder

# 方式一：内置经过实战验证的 Prompt（关键词提取推荐）
# 使用 prompt_template_3.py，包含结构化推理指令
prompt_builder = PromptBuilder.create_keyword_extraction_builder()

# 方式二：从 Schema 自动生成（适合自定义任务）
# 注意：使用此方式时，建议手动设置
# config.response_format = {"type": "json_object"} 以强制纯 JSON 输出
# prompt_builder = PromptBuilder.from_task(task)

# 方式三：完全自定义
# def my_generator(input_text: str, task) -> tuple:
#     return "你是提取专家...", f"提取：{input_text}"
# prompt_builder = PromptBuilder(custom_prompt_generator=my_generator)
```

> 完整用法见 [`examples/keyword_extraction/run.py`](examples/keyword_extraction/run.py)。

### Config — 调整行为

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

# response_format 默认为 None（允许 Prompt 模板输出混合文本）。
# 如果使用方式二（Schema 驱动），且希望模型只输出纯 JSON，需手动设置：
# config.response_format = {"type": "json_object"}
```

### RLRefineProcessor — 执行提取

```python
from core.processor import RLRefineProcessor

processor = RLRefineProcessor(config=config, task=task, prompt_builder=prompt_builder)

# 单条
result = processor.process_single(text="...", text_id="001")

# 批量（默认 10 线程并发）
results = processor.process_batch([
    {"id": "001", "describe": "..."},
    {"id": "002", "describe": "..."},
])
```

**执行流程：**
```
输入 → 预处理 → Prompt 生成 → LLM 调用 → 解析 JSON → Schema 验证 → 后处理 → 输出
                                       ↓ 失败时
                                 重试（第 3 次起加惩罚参数）
                                       ↓ 全部失败
                                 Jieba 兜底提取 → 输出
```

---

## RL 训练流程

### 训练流程总览

```
Qwen2.5-7B-Instruct（基座）
    │
    ▼ SFT  ── 通过示范数据教会推理模式
    │         （融合 LoRA：bash rl/merge_lora.sh）
    │
    ▼ DPO  ── 通过好/差对比教会质量偏好
    │         （融合 LoRA：bash rl/merge_lora.sh）
    │
    ▼ GRPO ── 以提取 F1 为奖励信号直接优化
    │         （融合 LoRA：bash rl/merge_lora.sh）
    │
    ▼
精炼后模型（系统性推理 + 准确提取）
```

> 每个 LoRA 训练阶段结束后，使用 `rl/merge_lora.sh` 将适配器权重合并到基座模型中，再进入下一阶段。

### 步骤 1：准备训练数据

**SFT 数据**：使用强模型（如 Qwen3-Max）生成高质量的推理 + 提取示范数据。可参考 `scripts/generate_data_qwen3-max.py` 脚本和 `prompts/prompt_template_3.py` 提示词模板。

Prompt 指示模型输出两个部分——推理部分（"思考"）和 JSON 结果：

```json
{
  "messages": [
    {"role": "system", "content": "（prompt_template_3.py 中的 prompt）"},
    {"role": "user", "content": "请分析以下评论：..."},
    {"role": "assistant", "content": "**思考**\n第一步：识别主体词... '屏幕'是产品属性词...\n\n**JSON输出**\n{\"keywords\": [[\"产品属性\", \"屏幕\", 0.95]]}"}
  ]
}
```

**DPO 数据**：使用 `scripts/generate_data_dpo.py` 脚本配合 `prompts/prompt_template_dpo.py` 提示词模板，生成 chosen/rejected 偏好对数据。

**GRPO 数据**：将 SFT 数据转为 GRPO 格式（移除 assistant 回复，只保留 prompt）：

```bash
python rl/convert_sft_to_grpo.py
```

> 关于 ms-swift 标准的 DPO 和 GRPO 训练数据格式要求，请参考 [ms-swift 自定义数据集文档](https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-dataset.html#dpo-orpo-cpo-simpo-rm)。

### 步骤 2：SFT 训练

```bash
bash rl/sft_finetune.sh
bash rl/merge_lora.sh    # 融合 LoRA 权重到基座模型
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | Qwen2.5-7B-Instruct | 基座模型 |
| `LORA_RANK` | `16` | LoRA 秩 |
| `LEARNING_RATE` | `1e-5` | 学习率 |
| `MAX_LENGTH` | `8192` | 最大序列长度 |

### 步骤 3：DPO 训练

```bash
bash rl/dpo_finetune.sh
bash rl/merge_lora.sh    # 融合 LoRA 权重
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `./output/sft/xxx-merged` | SFT 合并后模型 |
| `BETA` | `0.2` | KL 惩罚系数 |
| `LEARNING_RATE` | `5e-7` | 学习率（DPO 需要很小的值） |

### 步骤 4：GRPO 训练

```bash
bash rl/grpo_finetune.sh
bash rl/merge_lora.sh    # 融合 LoRA 权重
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `./output/dpo/xxx-merged` | DPO 合并后模型 |
| `BETA` | `0.01` | KL 惩罚系数（比 DPO 更小） |
| `NUM_GENERATIONS` | `8` | 每个 prompt 采样的输出数量 |
| `TEMPERATURE` | `0.9` | 采样温度 |

---

## 自定义任务

### 示例：情感分析

```python
# schema.py
def create_sentiment_task() -> ExtractionTask:
    schema = TaskSchema(name="sentiment", description="文本情感分析")
    schema.add_field(FieldDefinition(
        name="sentiment", type=FieldType.STRING, required=True,
        enum_values=["positive", "negative", "neutral"]
    ))
    schema.add_field(FieldDefinition(
        name="confidence", type=FieldType.FLOAT, required=True,
        min_value=0.0, max_value=1.0
    ))
    return ExtractionTask(schema=schema,
                          custom_rules=["置信度反映情感明确程度"])
```

```python
# run.py
task = create_sentiment_task()
config = Config(task_schema=task.schema, task=task)
config.enable_post_process = False
config.response_format = {"type": "json_object"}  # Schema 驱动方式下强制纯 JSON 输出

builder = PromptBuilder.from_task(task)
processor = RLRefineProcessor(config=config, task=task, prompt_builder=builder)
results = processor.process_batch([{"id": "1", "text": "东西很好！"}])
```

---

## 性能说明

### 单条推理延迟

使用 vLLM 本地部署时，单条推理通常需要 **5-25 秒**。这是自回归 LLM 推理的固有特性（逐 token 生成），不是框架本身的问题。

| 场景 | 近似耗时 |
|------|---------|
| 短文本 + 原始模型 | ~5 秒 |
| 长文本 + 原始模型 | ~8 秒 |
| 长文本 + RL 训练后模型 | ~20-25 秒 |

### 提升吞吐量

1. **使用批量处理** —— `process_batch()` 默认 10 线程并发，处理大数据集时吞吐量大幅提升
2. **调整 vLLM 参数** —— 增大 `scripts/run_vllm.sh` 中的 `--max-num-seqs` 和 `--gpu-memory-utilization`
3. **简单任务考虑更短的 Prompt** —— 内置关键词提取 Prompt 较为详尽；对于简单任务，Schema 驱动方式（方式二）配合 `response_format=json_object` 可减少输出量

---

## 目录结构

```
RLRefine/
├── rl/                                # RL 训练模块
│   ├── reward_builder.py              #   Schema 驱动的奖励函数（GRPO）
│   ├── convert_sft_to_grpo.py         #   SFT → GRPO 数据格式转换
│   ├── merge_lora.sh                  #   LoRA 权重融合脚本（每个训练阶段后使用）
│   ├── sft_finetune.sh                #   SFT 训练脚本（ms-swift）
│   ├── dpo_finetune.sh                #   DPO 训练脚本（ms-swift）
│   └── grpo_finetune.sh               #   GRPO 训练脚本（ms-swift + vLLM）
│
├── core/                              # 核心推理模块
│   ├── schema.py                      #   Schema 定义与验证
│   ├── config.py                      #   配置系统
│   ├── processor.py                   #   核心处理器（推理 + 重试 + 兜底）
│   ├── preprocess.py                  #   文本预处理
│   ├── post_process.py                #   关键词后处理
│   └── fallback.py                    #   Jieba 兜底（TF-IDF / TextRank）
│
├── prompts/                           # Prompt 模块
│   ├── prompt_builder.py              #   动态 Prompt 生成器
│   ├── prompt_template_3.py           #   关键词提取模板（含推理指令）
│   ├── prompt_template_4_short.py     #   关键词提取模板（短文本）
│   └── prompt_template_dpo.py         #   DPO 数据生成模板（chosen/rejected）
│
├── scripts/                           # 工具脚本
│   ├── run_vllm.sh                    #   vLLM 启动脚本
│   ├── generate_data_qwen3-max.py     #   SFT 数据生成（参考）
│   └── generate_data_dpo.py           #   DPO 数据生成（参考）
│
├── examples/
│   └── keyword_extraction/            #   完整可运行示例
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## 常见问题

### Q: 可以不使用 vLLM 吗？

本项目仅在 **vLLM + Qwen2.5-7B-Instruct** 上经过充分测试。其他 OpenAI 兼容后端理论上可用，但 `extra_body={"repetition_penalty": ...}` 是 vLLM 特有参数，OpenAI 官方 API 不支持。框架已对此做了自动容错处理（检测到不支持时移除该参数并重试）。

### Q: 模型输出被截断（finish_reason=length）？

增大 `config.max_token`（默认 1024，建议试 2048）。

### Q: 可以只用推理功能，不做 RL 训练吗？

可以。`rl/` 下的训练脚本完全可选。只需 `core/`、`prompts/`、`examples/` 即可运行推理。安装时可跳过 `ms-swift` 和 `vllm`。

### Q: 为什么单条推理速度慢？

自回归 LLM 推理是逐 token 生成的。RL 训练后的模型因为生成更详细的推理过程会更慢。详见 [性能说明](#性能说明)。

### Q: `response_format` 是什么？什么时候需要设置？

默认情况下 `response_format` 为 `None`，允许模型输出混合文本（推理 + JSON）。使用方式一（内置 Prompt 模板）时必须保持为 `None`，因为 Prompt 指示模型在 JSON 之前先输出推理过程。如果使用方式二（Schema 驱动），且希望强制纯 JSON 输出，需手动设置 `config.response_format = {"type": "json_object"}`。

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM 服务地址 |
| `VLLM_API_KEY` | `dummy` | API Key（vLLM 通常不验证） |
| `MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | 模型名称或本地路径 |
| `DEBUG` | `false` | 调试日志开关 |

---

## License

Apache 2.0
