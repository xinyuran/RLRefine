[English](README_EN.md) | 中文

# RLRefine

**Schema 驱动的结构化信息提取框架，支持 RL 训练增强**

> 定义一个 Schema，即可驱动 LLM 完成任意结构化提取任务。需要更高质量？用 RL 训练（SFT → DPO → GRPO）精炼模型的推理与提取能力。

---

## 效果展示

以中文电商评论关键词提取为例，对比 Qwen2.5-7B-Instruct 原始模型和 RL 训练后模型：

```text
终于收到我需要的宝贝了，东西很好，价美物廉，谢谢掌柜的！说实在，这是我购物以来来让我最满意的一次购物。
无论是掌柜的态度还是对物品，我都非常满意的。掌柜态度很专业热情，有问必答，回复也很快……再说宝贝，
正是我需要的，收到的时候包装完整，打开后让我惊喜的是，宝贝比我想象中的还要好！
```

| 维度 | 原始模型 | RL 训练后模型 |
|------|---------|--------------|
| 提取关键词数 | 6 个 | 8 个 |
| 推理结构 | 简单 Markdown 列表 | 系统化五步分析 |
| 置信度类型 | 字符串 `"0.95"`（错误） | 数字 `0.95`（正确） |
| 遗漏关键词 | "满意"、"好"、"包装" | 无 |

**原始模型**：`['宝贝', '态度', '回复', '价美物廉', '购物', '优惠']`

**RL 训练后**：`['满意', '好', '宝贝', '价美物廉', '态度', '回复', '包装', '优惠']`

<details>
<summary><b>点击展开：模型完整输出对比</b></summary>

#### 原始 Qwen2.5-7B-Instruct 输出（626 字符，耗时 ~8 秒）

```text
### 思考
1. **主体定位**：
   - 主体对象：宝贝、掌柜、态度、物品、包装。
2. **评价、描述定位**：
   - "东西很好"：主体“东西”，描述“好”。
   - "价美物廉"：主体“价美物廉”。
   - "这是我购物以来让我最满意的一次购物"：主体“购物”，描述“满意”。
   - "掌柜态度很专业热情"：主体“态度”，描述“专业热情”。
   - "有问必答，回复也很快"：主体“回复”，描述“快”。
   - "宝贝比我想象中的还要好"：主体“宝贝”，描述“好”。
3. **补充名词**：
   - “优惠”：与购物相关的重要名词。
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

问题：推理过程粗糙（简单列表），置信度为字符串类型 `"0.95"` 而非数字，遗漏了"满意"、"好"、"包装"等重要关键词。

#### RL 训练后模型输出（1,691 字符，耗时 ~23 秒）

```text
思考
首先通读整条评论，识别主体对象和评价描述。评论围绕一次购物体验展开，核心主体包括“宝贝”、“包装”、“掌柜的态度”、“回复速度”等。根据规则，需提取原子级关键词，且每个关键词≤4字，忠于原文，不可改写。

第一步：提取全局主体对象。文中多次提到“宝贝”，是核心商品主体；“包装”是物流/交付环节的关键属性；“掌柜”指代卖家或客服角色，属于服务主体；“回复”关联到客服响应效率。

第二步：逐句分析评价与描述：
- “东西很好” → “东西”可视为“宝贝”的同义泛称，但更精准的主体是“宝贝”；“很好”是正面评价，但“好”已足够表达，不过结合上下文，“东西很好”中的“好”可作为通用评价词保留。
- “价美物廉”是固定四字成语，符合规则，整体作为原子关键词。
- “最满意的一次购物” → “满意”是核心情绪词，需提取。
- “掌柜的态度很专业热情” → 拆出“态度”（对象）、“专业”（描述）、“热情”（描述）。
- “有问必答，回复也很快” → “回复”是服务行为，“快”是描述，但“回复快”中“快”需单独作为关键词；同时“回复”本身是重要服务属性，应保留。
- “不觉得烦” → 表达耐心，但无直接≤4字原子词，不提取。
- “认真回答” → “认真”是态度描述，可提取。
- “包装完整” → “包装”是对象，“完整”是状态描述，均≤4字，分别提取。
- “宝贝比我想象中的还要好” → 再次强调“宝贝”，“好”已覆盖。
- “竖起大拇指” → 情绪表达，但非具体关键词，不提取。
- “会再来的” → 表达复购意愿，但无具体新关键词。
- “给个优惠” → “优惠”是价格相关诉求，属有意义名词，应提取。

第三步：检查否定/问题场景——本评论无负面内容，无需处理否定结构。

第四步：过滤冗余与合并：
- “东西” vs “宝贝”：“宝贝”更具体，优先保留“宝贝”。
- “好”出现多次，但作为通用评价词，重要性高。
- “满意”出现两次（购物、宝贝），重要性高。
- “专业”“热情”“认真”均为对“态度”的具体描述，需分别提取。
- “回复”“包装”“优惠”均为关键属性名词，必须提取。
- “价美物廉”作为整体成语，不可拆分，直接保留。

第五步：控制数量≤15，并按重要性排序：
- 最高频/强情感词：“满意”“好”“宝贝”“价美物廉”优先；
- 服务维度：“态度”“回复”“包装”“优惠”；
- 具体描述词：“专业”“热情”“认真”。

最终确定15个以内关键词，确保每条记录仅一个关键词，且均≤4字，忠于原文。

{
  "keywords": [
    ["评论多次强调整体购物体验满意，'满意'为核心情绪词", "满意", 0.95],
    ["'东西很好'中提取通用正面评价词'好'", "好", 0.90],
    ["全文核心商品主体，多次提及'宝贝'", "宝贝", 0.88],
    ["固定四字成语'价美物廉'直接引用，整体作为原子关键词", "价美物廉", 0.85],
    ["'掌柜的态度'中提取服务对象词'态度'", "态度", 0.82],
    ["'回复也很快'中提取服务行为关键词'回复'", "回复", 0.80],
    ["'包装完整'中提取交付属性关键词'包装'", "包装", 0.78],
    ["'给个优惠'中提取价格相关诉求名词'优惠'", "优惠", 0.75],
    ["'态度很专业'中提取描述词'专业'", "专业", 0.72],
    ["'态度很热情'中提取描述词'热情'", "热情", 0.70],
    ["'认真回答'中提取态度描述词'认真'", "认真", 0.68],
    ["'东西很好'中'好'作为通用正面评价，需单独提取", "好", 0.65],
    ["'包装完整'中'完整'描述包装状态，需提取", "完整", 0.62],
    ["'回复也很快'中'快'描述回复速度，需提取", "快", 0.60]
  ]
}
```

改进：系统化提示词中要求的推理步骤，置信度为正确的数字类型 `0.95`，每条关键词附带提取理由，无遗漏，同时增强可解释性，思考过程也更完善、合理。

</details>

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

---

## 架构总览

RLRefine 由两个独立模块组成：

| | 模块一：Schema 驱动推理 | 模块二：RL 训练增强 |
|---|---|---|
| **用途** | 定义 Schema，直接用 LLM 做结构化提取 | 通过 RL 训练提升模型的推理和提取质量 |
| **使用成本** | 低——定义 Schema + 部署 vLLM | 高——需要 GPU 训练环境 |
| **依赖** | `openai`、`tqdm`、`jieba`、`vllm` | `ms-swift` |
| **是否必需** | 是 | 否（可选增强） |

**Schema 是整个框架的核心**——定义一次，推理和训练共同使用：

```
┌───────────────────────────────────────────────┐
│                    Schema                     │
│     （Python 代码定义任务结构，一次定义共用）    │
└──────────────────┬────────────────────────────┘
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
┌──────────────┐      ┌──────────────────┐
│  模块一：推理  │      │  模块二：RL 训练   │
│              │      │                  │
│ PromptBuilder│      │  Reward Builder  │
│ → LLM 调用   │      │  SFT → DPO → GRPO│
│ → Schema 验证 │      │  → 精炼后模型     │
│ → 后处理      │      │                  │
└──────────────┘      └──────────────────┘
```

---

## 模块一：Schema 驱动推理

> 低使用成本：定义 Schema → 生成 Prompt → 调用 LLM → 结构化输出

### 快速开始

**1. 启动 vLLM 服务**

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# 或使用项目脚本
bash scripts/run_vllm.sh 0 8000    # GPU 0, 端口 8000
```

**2. 配置环境变量**

```bash
cp .env.example .env
# 编辑 .env：
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=dummy
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```

**3. 运行示例**

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

### 核心 API

#### 1. Schema — 定义任务结构

`TaskSchema` 是框架的核心，定义提取任务的字段结构和约束：

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

支持的字段类型：`STRING`、`INTEGER`、`FLOAT`、`BOOLEAN`、`ARRAY`、`OBJECT`、`ARRAY_OF_OBJECTS`

每个字段可设置：`required`、`min_value`/`max_value`、`enum_values`、`min_length`/`max_length`、`pattern` 等约束。

#### 2. PromptBuilder — 生成 Prompt

提供三种方式，从开箱即用到完全自定义：

```python
from prompts.prompt_builder import PromptBuilder

# 方式一：内置模板（关键词提取，经过实战验证）
prompt_builder = PromptBuilder.create_keyword_extraction_builder()

# 方式二：从 Schema 自动生成（适合自定义任务）
prompt_builder = PromptBuilder.from_task(task)

# 方式三：完全自定义 Prompt 生成函数
def my_generator(input_text: str, task) -> tuple:
    return "你是提取专家...", f"提取：{input_text}"
prompt_builder = PromptBuilder(custom_prompt_generator=my_generator)
```

> **方式一** 使用 `prompt_template_3.py`，指示模型先输出推理过程再输出 JSON，`config.response_format` 须保持 `None`。
> **方式二** 根据 Schema 自动生成 Prompt，建议设置 `config.response_format = {"type": "json_object"}` 强制纯 JSON 输出。

#### 3. Config — 调整行为

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

#### 4. Processor — 执行提取

```python
from core.processor import RLRefineProcessor

processor = RLRefineProcessor(config=config, task=task, prompt_builder=prompt_builder)

# 单条处理
result = processor.process_single(text="...", text_id="001")

# 批量处理（默认 10 线程并发）
results = processor.process_batch([
    {"id": "001", "describe": "..."},
    {"id": "002", "describe": "..."},
])
```

**执行流程：**
```
输入 → Prompt 生成 → LLM 调用 → JSON 解析 → Schema 验证 → 后处理 → 输出
                                  ↓ 失败
                            重试（第 3 次起加惩罚参数）
                                  ↓ 全部失败
                            Jieba TF-IDF 兜底 → 输出
```

### 自定义任务示例：情感分析

只需定义新的 Schema，即可将框架用于完全不同的提取任务：

```python
from core.schema import TaskSchema, FieldDefinition, FieldType, ExtractionTask
from core.config import Config
from core.processor import RLRefineProcessor
from prompts.prompt_builder import PromptBuilder

# 1. 定义 Schema
schema = TaskSchema(name="sentiment", description="文本情感分析")
schema.add_field(FieldDefinition(
    name="sentiment", type=FieldType.STRING, required=True,
    enum_values=["positive", "negative", "neutral"]
))
schema.add_field(FieldDefinition(
    name="confidence", type=FieldType.FLOAT, required=True,
    min_value=0.0, max_value=1.0
))
task = ExtractionTask(schema=schema, custom_rules=["置信度反映情感明确程度"])

# 2. 配置（Schema 驱动方式，强制纯 JSON 输出）
config = Config(task_schema=task.schema, task=task)
config.enable_post_process = False
config.response_format = {"type": "json_object"}

# 3. 生成 Prompt 并执行
builder = PromptBuilder.from_task(task)
processor = RLRefineProcessor(config=config, task=task, prompt_builder=builder)
results = processor.process_batch([{"id": "1", "text": "东西很好！"}])
```

---

## 模块二：RL 训练增强

> 高使用成本：需要 GPU 训练环境、`ms-swift`、`vllm`

通过三阶段 RL 训练，教会模型在提取前进行系统性推理，显著提升提取质量。

### 训练流程

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

> 每个阶段训练结束后，使用 `rl/merge_lora.sh` 将 LoRA 权重合并到基座模型，再进入下一阶段。

### 步骤 1：准备训练数据

**SFT 数据**：使用强模型（如 Qwen3-Max）生成高质量推理 + 提取示范。参考 `rl/generate_data_qwen3-max.py` 和 `prompts/prompt_template_3.py`。

```json
{
  "messages": [
    {"role": "system", "content": "（prompt_template_3.py 中的 prompt）"},
    {"role": "user", "content": "请分析以下评论：..."},
    {"role": "assistant", "content": "**思考**\n第一步：识别主体词...\n\n**JSON输出**\n{\"keywords\": [...]}"}
  ]
}
```

**DPO 数据**：使用 `rl/generate_data_dpo.py` 配合 `prompts/prompt_template_dpo.py`，生成 chosen/rejected 偏好对。

**GRPO 数据**：将 SFT 数据转为 GRPO 格式（移除 assistant 回复，只保留 prompt）：

```bash
python rl/convert_sft_to_grpo.py
```

> 数据格式要求参考 [ms-swift 自定义数据集文档](https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-dataset.html#dpo-orpo-cpo-simpo-rm)。

### 步骤 2：SFT 训练

```bash
bash rl/sft_finetune.sh
bash rl/merge_lora.sh
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
bash rl/merge_lora.sh
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `./output/sft/xxx-merged` | SFT 合并后模型 |
| `BETA` | `0.2` | KL 惩罚系数 |
| `LEARNING_RATE` | `5e-7` | 学习率（DPO 需要很小的值） |

### 步骤 4：GRPO 训练

```bash
bash rl/grpo_finetune.sh
bash rl/merge_lora.sh
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `./output/dpo/xxx-merged` | DPO 合并后模型 |
| `BETA` | `0.01` | KL 惩罚系数（比 DPO 更小） |
| `NUM_GENERATIONS` | `8` | 每个 prompt 采样数 |
| `TEMPERATURE` | `0.9` | 采样温度 |

### 奖励设计

准确率权重占 50%——**提取质量是最终目标**，格式和推理约束是手段：

| 维度 | 权重 | 作用 |
|------|------|------|
| 准确率（F1） | **50%** | 核心提取质量 |
| Schema 验证 | 20% | 输出正确性 |
| 格式检查 | 20% | JSON 可靠性 |
| 思考质量 | 10% | 推理鲁棒性 |
| 幻觉惩罚 | -0.1/词 | 防止编造 |

---

## 目录结构

```
RLRefine/
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
│   └── prompt_template_dpo.py         #   DPO 数据生成模板
│
├── rl/                                # RL 训练模块
│   ├── reward_builder.py              #   Schema 驱动的奖励函数（GRPO）
│   ├── generate_data_qwen3-max.py     #   SFT 数据生成（参考）
│   ├── generate_data_dpo.py           #   DPO 数据生成（参考）
│   ├── convert_sft_to_grpo.py         #   SFT → GRPO 数据格式转换
│   ├── merge_lora.sh                  #   LoRA 权重融合脚本
│   ├── sft_finetune.sh                #   SFT 训练脚本（ms-swift）
│   ├── dpo_finetune.sh                #   DPO 训练脚本（ms-swift）
│   └── grpo_finetune.sh               #   GRPO 训练脚本（ms-swift + vLLM）
│
├── scripts/                           # 工具脚本
│   └── run_vllm.sh                    #   vLLM 启动脚本
│
├── examples/
│   └── keyword_extraction/            #   完整可运行示例
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## License

Apache 2.0
