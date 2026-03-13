# StructAlign: 结构化数据抽取与对齐框架

> **Stable Structured Output via RLHF (SFT -> DPO -> GRPO)**
>
> 一个基于强化学习（RL）的端到端框架，专注于提升大模型输出结构化数据（JSON）的稳定性与准确率。

## 📖 项目简介 (Introduction)

在实际的 LLM 生产环境中，**“不仅要答对，还要格式对”** 是最大的挑战。StructAlign 并不重复造轮子（底层基于 ms-swift 等成熟库），而是提供了一套经过实战验证的**工程化解决方案**。

本项目源于电商评论关键词抽取的实战需求，抽象出了一套通用的 **SFT (有监督微调) -> DPO (直接偏好优化) -> GRPO (组相对策略优化)** 训练流水线，配合**鲁棒的混合推理引擎**，能够显著降低 LLM 输出 JSON 时的格式错误率（Format Error Rate）和幻觉率。

### 核心价值 (Core Value)
*   🎯 **专注结构化稳定性**：利用 GRPO 强化学习算法，将“JSON 格式正确性”作为 Reward 信号，直接优化模型的输出行为。
*   🔄 **数据自举流水线**：内置从 SFT 数据自动生成 DPO/GRPO 偏好数据的工具（Self-Instruct），解决 RL 训练“缺数据”的难题。
*   🛡️ **工业级推理引擎**：集成了 `Retry-with-Penalty`（带惩罚重试）、`Rule-Based Fallback`（规则兜底）等多级容错机制，保障线上高可用。

## 🏗️ 架构概览 (Architecture)

项目包含三个核心子系统：**数据合成 (Data Synthesis)**、**RL 训练 (RL Training)** 和 **推理引擎 (Inference Engine)**。

```mermaid
graph TD
    subgraph "Phase 1: 数据流 (Data Pipeline)"
        Raw[原始文本] --> Pre[预处理 (Preprocess)]
        Pre --> SFT_Data[SFT 数据集]
        SFT_Data --> |Teacher Model| Gen[数据合成 (Generate Data)]
        Gen --> DPO_Data[DPO 偏好数据 (Chosen/Rejected)]
        Gen --> GRPO_Data[GRPO 提示词数据 (Prompts)]
    end

    subgraph "Phase 2: 训练流 (Training Pipeline)"
        Base[基座模型] --> |SFT| SFT_Model[SFT 模型]
        SFT_Model --> |DPO| DPO_Model[DPO 对齐模型]
        DPO_Model --> |GRPO + Format Reward| Final_Model[最终 RL 模型]
    end

    subgraph "Phase 3: 推理流 (Inference Engine)"
        Input --> LLM[LLM 推理]
        LLM --> |JSON Parse Error| Retry[带惩罚参数重试]
        Retry --> |Still Fail| Fallback[规则/Jieba 兜底]
        Fallback --> Output[最终结构化输出]
    end
```

## 📂 模块说明 (Modules)

### 1. 强化学习流水线 (`/rl`)
这是本框架的核心，通过三阶段训练逐步提升模型能力。

*   **SFT (Supervised Fine-Tuning)**: 注入领域知识（如：什么是“商品关键词”）。
*   **DPO (Direct Preference Optimization)**: 
    *   **实现**: `rl/dpo_finetune.sh`
    *   **作用**: 抑制模型的幻觉和啰嗦，让模型更倾向于输出简洁的 JSON。
    *   **数据生成**: `rl/generate_data_dpo.py` 利用高阶模型（如 Qwen-Max）生成对比数据，自动构建 `<chosen>` 和 `<rejected>` 样本。
*   **GRPO (Group Relative Policy Optimization)**:
    *   **实现**: `rl/grpo_finetune.sh`
    *   **作用**: **这是解决 JSON 不稳定的关键**。通过采样多组输出，计算 Group 内部的优势函数。
    *   **Reward Function**: 定义了 `robust_keyword_reward`，对 JSON 解析失败、字段缺失进行强惩罚，倒逼模型学会“严格遵守格式”。

### 2. 鲁棒推理引擎 (`/core`)
仅仅训练好模型是不够的，线上环境需要极高的稳定性。

*   **Processor (`core/processor.py`)**: 
    *   封装了完整的推理生命周期。
    *   **动态重试策略**: 首轮使用常规参数 -> 失败后增加 `frequency_penalty` 和 `repetition_penalty` 再次尝试。
    *   **多级兜底**: 当 LLM 彻底失效时，自动降级为 `jieba` 分词或正则提取，确保系统永远有返回值。
*   **Prompt Templates (`core/prompt_template_*.py`)**: 
    *   针对长文本/短文本自动切换不同的 Prompt 策略，优化 Token 消耗与提取效果。

### 3. 数据工程 (`/core/preprocess.py`)
*   提供了一套针对中文社交媒体/电商评论的清洗工具（去除 Emoji、乱码、HTML 标签等），保证输入模型的 Data Quality。

## 🚀 快速开始 (Quick Start)

### 环境准备
```bash
# 依赖安装
pip install ms-swift modelscope vllm
pip install -r requirements.txt
```

### 步骤 1: 数据准备与合成
将你的原始数据转换为 SFT 格式，然后生成 RL 训练所需的数据：
```bash
# 1. 转换 SFT 数据为 GRPO 格式 (只保留 Prompt)
python rl/convert_sft_to_grpo.py

# 2. (可选) 生成 DPO 数据
python rl/generate_data_dpo.py
```

### 步骤 2: 启动训练
建议按照 SFT -> DPO -> GRPO 的顺序进行训练。

```bash
# 启动 GRPO 训练 (需配置 accelerate)
cd rl
chmod +x grpo_finetune.sh
./grpo_finetune.sh
```
*注意：`grpo_finetune.sh` 中需修改 `MODEL_PATH` 指向你的 SFT 或 DPO 模型路径。*

### 步骤 3: 推理与服务
使用 `processor` 进行单条或批量预测：

```python
from core.processor import KeywordExtractor

extractor = KeywordExtractor(vllm_base_url="http://localhost:8000/v1")
result = extractor.process_single("这件衣服面料很舒服，但是尺码偏小", text_id="1001")
print(result)
# Output: {'id': '1001', 'keywords': ['面料舒服', '尺码偏小']}
```

## 🛠️ 自定义指南 (Customization)

*   **修改提取目标**: 修改 `core/prompt_template_3.py` 中的 System Prompt，定义你需要的 JSON Schema（如提取实体、情感分析等）。
*   **自定义 Reward**: 在 `rl/reward_functions.py` 中实现针对你特定 Schema 的校验逻辑（例如：检查是否包含特定字段、数值范围是否合法）。

## 📄 License
Apache 2.0
