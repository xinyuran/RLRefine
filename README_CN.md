[English](README_EN.md) | 中文

# RLRefine

**基于 RL 训练增强的 LLM 结构化信息提取实践**

> RLRefine 是一个将 RL 训练（SFT → DPO → GRPO）应用于 LLM 结构化信息提取的完整实践。以中文电商评论关键词提取为核心场景，展示了从 Schema 定义、Prompt 工程、多阶段 RL 训练到奖励函数设计的全流程。框架本身具备通用性——只需更换 Schema 定义即可适配其他提取任务。

---

## 模型权重

训练好的模型权重已上传至 HuggingFace：[https://huggingface.co/xinyuran/Qwen2.5-7B-RLRefine](https://huggingface.co/xinyuran/Qwen2.5-7B-RLRefine)

---

## 核心亮点

- **完整的 RL 训练流水线**：SFT → DPO → GRPO 三阶段渐进式训练，基于 Qwen2.5-7B-Instruct 在 2×H100 上完成全流程
- **Schema 驱动的通用设计**：定义一次 Schema，推理和训练共同使用；更换 Schema 即可适配情感分析、实体提取等其他任务
- **可解释的奖励函数**：五维度复合奖励（F1 50% + Schema验证 20% + 格式 20% + 思考质量 10% + 幻觉惩罚），附完整设计文档
- **端到端可复现**：提供训练脚本、数据样本、训练日志和评测方案

---

## 效果展示

以中文电商评论关键词提取为例，对比 Qwen2.5-7B-Instruct 原始模型和 RL 训练后模型：

```text
终于收到我需要的宝贝了，东西很好，价美物廉，谢谢掌柜的！说实在，这是我购物以来让我最满意的一次购物。
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

#### 原始 Qwen2.5-7B-Instruct 输出（726 字符，耗时 ~8 秒）

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
    ["掌柜给我一优惠哦", "优惠", "0.70"],
    ["东西很好", "东西", "0.65"]
  ]
}
```

问题：推理过程粗糙（简单列表），置信度为字符串类型 `"0.95"` 而非数字，遗漏了"满意"、"好"、"包装"等重要关键词。

#### RL 训练后模型输出（1,691 字符，耗时 ~23 秒）

```text
思考
首先通读整条评论，识别主体对象和评价描述。评论围绕一次购物体验展开，核心主体包括"宝贝"、"包装"、"掌柜的态度"、"回复速度"等。根据规则，需提取原子级关键词，且每个关键词≥4字，忠于原文，不可改写。

第一步：提取全局主体对象。文中多次提到"宝贝"，是核心商品主体；"包装"是物流/交付环节的关键属性；"掌柜"指代卖家或客服角色，属于服务主体；"回复"关联到客服响应效率。

第二步：逐句分析评价与描述：
- "东西很好" → "东西"可归于"宝贝"的同义泛称，但更精准的主体是"宝贝"；"很好"是正面评价，但"好"已足够表达，不过结合上下文，"东西很好"中的"好"可作为通用评价词保留。
- "价美物廉"是固定四字成语，符合规则，整体作为原子关键词。
- "最满意的一次购物" → "满意"是核心情绪词，需提取。
- "掌柜的态度很专业热情" → 拆出"态度"（对象）、"专业"（描述）、"热情"（描述）。
- "有问必答，回复也很快" → "回复"是服务行为，"快"是描述，但"回复快"→"快"需单独作为关键词；同时"回复"本身是重要服务属性，应保留。
- "包装完整" → "包装"是对象，"完整"是状态描述，坚团≥4字，分别提取。
- "宝贝比我想象中的还要好" → 再次强调"宝贝"，"好"已被覆盖。
- "给个优惠" → "优惠"是价格相关诉求，属有意义名词，应提取。

第三步：检查是否含否定场景——本评论无负面内容，无需处理否定结构。

第四步：过滤冗余与合并：
- "东西" vs "宝贝"："宝贝"更具体，优先保留"宝贝"。
- "好"出现多次，但作为通用评价词，重要性高。
- "满意"出现两次（购物、宝贝），重要性高。
- "专业""热情""认真"均为对"态度"的具体描述，需分别提取。
- "回复""包装""优惠"均为关键属性名词，必须提取。
- "价美物廉"作为整体成语，不可拆分，直接保留。

第五步：控制数量≤5，并按重要性排序：
- 最高频/强情感词："满意""好""宝贝""价美物廉"优先；
- 服务维度："态度""回复""包装""优惠"；
- 具体描述词："专业""热情""认真"。

最终确定5个以内关键词，确保每条记录仅一个关键词，且坚团≥4字，忠于原文。
{
  "keywords": [
    ["评论多次强调整体购物体验满意，'满意'为核心情绪词", "满意", 0.95],
    ["'东西很好'中提取通用正面评价词'好'", "好", 0.90],
    ["全文核心商品主体，多次提及'宝贝'", "宝贝", 0.88],
    ["固定四字成语'价美物廉'直接引用，整体作为原子关键词", "价美物廉", 0.85],
    ["'掌柜的态度'中提取服务对象词'态度'", "态度", 0.82],
    ["'回复也很快'中提取服务行为关键词'回复'", "回复", 0.80],
    ["'包装完整'中提取交付属性关键词'包装'", "包装", 0.78],
    ["'给个优惠'中提取价格相关诉求名词'优惠'", "优惠", 0.75]
  ]
}
```

改进：系统化提示词中要求的推理步骤，置信度为正确的数字类型 `0.95`，每条关键词附带提取理由，无遗漏，同时加强可解释性，思考过程也更完善、合理。

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
| **是否必需** | 是 | 否（可选加强） |

**Schema 是整个框架的核心**——定义一次，推理和训练共同使用：

```
+-----------------------------------------------+
|                    Schema                     |
|   （Python 代码定义任务结构，一次定义共用）    |
+------------------+----------------------------+
                   |
       +-----------+-----------+
       v                       v
+--------------+      +------------------+
| 模块一：推理 |      | 模块二：RL 训练   |
|              |      |                  |
| PromptBuilder|      | Reward Builder   |
| -> LLM 调用  |      | SFT -> DPO -> GRPO|
| -> Schema 验证|     | -> 精炼后模型     |
| -> 后处理    |      |                  |
+--------------+      +------------------+
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

`TaskSchema` 是框架的核心，定义提取任务的字段结构和约束条件：

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
    custom_rules=["关键词必须来自原文，不可捏造"]
)
```

**更换 Schema 即可适配其他任务**，无需修改推理代码：

```python
# 情感分析
schema = TaskSchema(name="sentiment", description="评论情感分析")
schema.add_field(FieldDefinition(name="sentiment", type=FieldType.STRING, ...))
schema.add_field(FieldDefinition(name="score", type=FieldType.NUMBER, ...))

# 实体提取
schema = TaskSchema(name="ner", description="命名实体识别")
schema.add_field(FieldDefinition(name="entities", type=FieldType.ARRAY_OF_OBJECTS, ...))
```

#### 2. PromptBuilder — 生成 Prompt

```python
from prompts.prompt_builder import PromptBuilder

# 方式一：使用内置模板（关键词提取，经过验证）
prompt_builder = PromptBuilder.create_keyword_extraction_builder()

# 方式二：从 Schema 自动生成（适用于自定义任务）
prompt_builder = PromptBuilder.from_task(task)

# 方式三：完全自定义 Prompt 生成器
def my_generator(input_text: str, task) -> tuple:
    return "你是一名提取专家...", f"请提取：{input_text}"
prompt_builder = PromptBuilder(custom_prompt_generator=my_generator)
```

#### 3. Processor — 执行提取

```python
from core.processor import RLRefineProcessor

processor = RLRefineProcessor(config=config, task=task, prompt_builder=prompt_builder)

# 单条处理
result = processor.process_single(text="...", text_id="001")

# 批量处理（默认 10 个并发线程）
results = processor.process_batch([
    {"id": "001", "describe": "..."},
    {"id": "002", "describe": "..."},
])
```

**处理流程：**
```
输入 -> Prompt -> LLM 调用 -> JSON 解析 -> Schema 验证 -> 后处理 -> 输出
                                | 失败
                          重试（第3次启用惩罚参数）
                                | 全部失败
                          Jieba TF-IDF 降级 -> 输出
```

---

## 模块二：RL 训练增强

> 高使用成本：需要 GPU 训练环境、`ms-swift`、`vllm`

三阶段 RL 训练教会模型在提取前进行系统化推理。

### 训练流水线

```
Qwen2.5-7B-Instruct（基础模型）
    |
    v SFT  -- 通过示范数据教会模型推理模式
    |
    v DPO  -- 通过好/坏样本对教会模型质量偏好
    |
    v GRPO -- 以提取 F1 为奖励信号直接优化
    |
    v
精炼后模型（系统化推理 + 准确提取）
```

### 训练配置摘要

| 阶段 | 学习率 | LoRA Rank | Epochs | 训练时长 | 峰值显存 |
|------|--------|-----------|--------|---------|---------|
| SFT | 1e-5 | 16 | 5 | ~1h 9min | 77.4 GiB |
| DPO | 5e-7 | 8 | 1 | -- | -- |
| GRPO | 5e-7 | 8 | 1 | ~4h 35min | 66.7 GiB |

硬件：2× NVIDIA H100 80GB HBM3

> 完整训练细节、损失曲线和工程记录：[docs/training_details.md](docs/training_details.md)

### 奖励设计摘要

准确性权重 50%——**提取质量是最终目标**：

| 维度 | 权重 | 作用 |
|------|------|------|
| 准确性（F1） | **50%** | 核心提取质量 |
| Schema 验证 | 20% | 输出结构正确性 |
| 格式检查 | 20% | JSON 可靠性 |
| 思考质量 | 10% | 推理鲁棒性 |
| 幻觉惩罚 | -0.1/词 | 防止捏造内容 |

> 奖励函数详细设计和 Reward Hacking 分析：[docs/reward_design.md](docs/reward_design.md)

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [docs/reward_design.md](docs/reward_design.md) | 奖励函数设计：五维度权重、评分规则、Reward Hacking 分析 |
| [docs/training_details.md](docs/training_details.md) | 训练细节：硬件、各阶段配置、损失曲线、工程问题 |
| [docs/evaluation.md](docs/evaluation.md) | 评测：实际测试结论与已知局限 |
| [examples/sample_data/](examples/sample_data/) | 训练数据样本（SFT / DPO / GRPO 格式） |

---

## 项目结构

```
RLRefine/
+-- core/                              # 核心推理模块
|   +-- schema.py                      #   Schema 定义与验证
|   +-- config.py                      #   配置系统
|   +-- processor.py                   #   核心处理器（推理 + 重试 + 降级）
|   +-- preprocess.py                  #   文本预处理
|   +-- post_process.py                #   关键词后处理
|   +-- fallback.py                    #   Jieba 降级（TF-IDF / TextRank）
+-- prompts/                           # Prompt 模块
|   +-- prompt_builder.py              #   动态 Prompt 生成器
|   +-- prompt_template_3.py           #   关键词提取模板（含推理）
+-- rl/                                # RL 训练模块
|   +-- reward_builder.py              #   Schema 驱动的奖励函数（GRPO）
|   +-- sft_finetune.sh                #   SFT 训练脚本
|   +-- dpo_finetune.sh                #   DPO 训练脚本
|   +-- grpo_finetune.sh               #   GRPO 训练脚本（ms-swift + vLLM）
|   +-- train_logs/                    #   训练日志
+-- docs/                              # 详细文档
|   +-- reward_design.md               #   奖励函数设计
|   +-- training_details.md            #   训练细节与工程记录
|   +-- evaluation.md                  #   评测方案与结果
+-- examples/
|   +-- keyword_extraction/            #   完整可运行示例
|   +-- sample_data/                   #   训练数据样本
+-- .env.example
+-- requirements.txt
+-- README.md
```

---

## License

MIT
