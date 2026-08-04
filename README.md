# RLRefine

[English](README_EN.md) | 简体中文

面向大语言模型的 Schema 驱动结构化信息抽取工具包。

RLRefine 将任务定义、推理约束、后训练奖励和输出校验统一到同一份 Schema 与语义
合约中，帮助模型稳定输出可解析、可验证的结构化结果。仓库以中文电商评论关键词抽取
为主要示例，同时提供客服工单路由示例，展示如何在不重写核心流程的情况下适配新的
输出结构。

## 核心能力

- **Schema 驱动**：用字段类型、枚举、长度和嵌套结构定义任务输出。
- **训练与推理对齐**：SFT、DPO、GRPO 和在线推理共享结构化目标。
- **语义合约校验**：在 JSON Schema 之外检查来源忠实、关键词长度、数量和重复项。
- **可靠性处理**：支持重试、确定性修复、降级提取和失败状态区分。
- **OpenAI 兼容接口**：可连接 vLLM 等 OpenAI-compatible 推理服务。
- **可扩展任务**：关键词列表与固定字段分类任务使用同一套核心抽象。

## 工作流程

```text
任务 Schema
    │
    ├── Prompt 与 JSON Schema
    ├── SFT / DPO / GRPO 训练目标
    └── 推理结果校验
             │
             ├── 合法结果
             ├── 可确定性修复的结果
             └── 失败或需人工复核的结果
```

RLRefine 明确区分三个层次：

1. 请求是否成功完成；
2. 输出是否满足 JSON Schema；
3. 输出是否满足任务语义，例如关键词必须来自原文。

这种区分可以避免把“JSON 能解析”误认为“业务结果可信”。

## 快速开始

基础安装要求 Python 3.10+，不需要 GPU、模型权重或外部 API：

```bash
git clone https://github.com/xinyuran/RLRefine.git
cd RLRefine
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m scripts.contract_quickstart
```

Windows PowerShell 使用：

```powershell
.venv\Scripts\Activate.ps1
```

quickstart 会在 CPU 上验证关键词抽取和客服工单路由两种 Schema：

```json
{
  "keyword_contract": {"valid": true, "errors": []},
  "routing_schema": {"valid": true, "errors": []}
}
```

运行公开测试：

```bash
python -m unittest discover -v tests
```

## 连接模型服务

本项目三阶段后训练流程产出的公开合并模型可从 Hugging Face 获取：
[`xinyuran/Qwen2.5-7B-RLRefine`](https://huggingface.co/xinyuran/Qwen2.5-7B-RLRefine)。
该模型面向中文电商评论关键词抽取，可用于体验和复现下方的推理流程。

安装 GPU 训练与服务依赖：

```bash
python -m pip install -r requirements-training.txt
```

启动一个 OpenAI-compatible vLLM 服务：

```bash
bash scripts/run_vllm.sh 0 8001 xinyuran/Qwen2.5-7B-RLRefine
```

复制并修改环境变量：

```bash
cp examples/keyword_extraction/.env.example examples/keyword_extraction/.env
python examples/keyword_extraction/run.py
```

核心推理入口是 `core.processor.RLRefineProcessor`。任务 Schema 示例位于
`examples/keyword_extraction/schema.py` 和 `examples/intent_routing/schema.py`。

## 后训练配置

已验证流程以 Qwen2.5-7B-Instruct 为基座，通过 LoRA 依次完成监督微调、偏好优化和
基于奖励的强化学习。

| 阶段 | 主要目标 | LoRA rank / alpha | 学习率 | Epoch | 关键参数 |
|---|---|---:|---:|---:|---|
| SFT | 学习输出格式与抽取任务 | 16 / 64 | `1e-5` | 5 | 最大长度 8192 |
| DPO | 学习高低质量答案偏好 | 8 / 32 | `5e-7` | 1 | `beta=0.2` |
| GRPO | 直接优化结构与抽取奖励 | 8 / 16 | `5e-7` | 1 | `beta=0.01`，每题采样 4 个回答 |

主要训练环境：

| 项目 | 配置 |
|---|---|
| GPU | 2 × NVIDIA H100 80GB |
| 训练精度 | BF16 |
| 训练框架 | ms-swift 3.11.2 |
| 推理与生成 | vLLM 0.13.0 |
| 并行与显存优化 | DeepSpeed ZeRO-2、vLLM colocate |
| 基座模型 | Qwen2.5-7B-Instruct |

训练脚本位于 `rl/`。默认路径仅为示例，使用者需要提供自己的模型与数据。

## 直观效果对比

下面是一条真实保存的代表性样例。基座模型与完成 SFT、DPO、GRPO 三阶段后训练的模型
使用同一条评论、相同推理入口和相同后处理流程。该样例用于直观解释输出变化，不代替
后续的整体评测指标。

评论节选：

> 终于收到我需要的宝贝了，东西很好，价美物廉，谢谢掌柜的！说实在，这是我购物以来
> 让我最满意的一次购物。无论是掌柜的态度还是对物品，我都非常满意。掌柜态度很专业
> 热情，有问必答，回复也很快……收到的时候包装完整，宝贝比我想象中的还要好！下次
> 需要的时候我还会再来，到时候麻烦掌柜给个优惠。

| 对比项 | Qwen2.5-7B-Instruct 基座模型 | SFT → DPO → GRPO 最终模型 |
|---|---|---|
| 统一后处理后的关键词 | `宝贝、态度、回复、价美物廉、购物、优惠` | `满意、好、宝贝、价美物廉、态度、回复、包装、优惠` |
| 关键词数量 | 6 | 8 |
| 关键信息覆盖 | 遗漏“满意”“好”“包装” | 覆盖情绪、商品、服务与包装信息 |
| 原始置信度类型 | 字符串，例如 `"0.95"` | JSON 数字，例如 `0.95` |
| 主要结构问题 | 重复“宝贝”，且置信度类型错误 | 输出可解析，重复项由统一后处理去除 |

最终模型补回了强情绪词“满意”、评价词“好”和交付属性“包装”，同时不再保留较泛化的
“购物”。这类单样本差异说明模型行为如何变化；是否形成稳定收益仍以完整评测集结果为
准。

<details>
<summary><b>查看代表性原始 JSON 片段</b></summary>

基座模型的原始输出包含字符串置信度和重复关键词：

```json
{
  "keywords": [
    ["宝贝收到后包装完整", "宝贝", "0.95"],
    ["宝贝比我想象中的还要好", "宝贝", "0.95"],
    ["掌柜态度很专业热情", "态度", "0.90"],
    ["这是我购物以来让我最满意的一次购物", "购物", "0.75"]
  ]
}
```

完成 SFT、DPO、GRPO 后，模型能直接输出数字置信度，并提取此前遗漏的信息：

```json
{
  "keywords": [
    ["评论多次强调整体购物体验满意，'满意'为核心情绪词", "满意", 0.95],
    ["'东西很好'中提取通用正面评价词'好'", "好", 0.90],
    ["全文核心商品主体，多次提及'宝贝'", "宝贝", 0.88],
    ["'包装完整'中提取交付属性关键词'包装'", "包装", 0.78]
  ]
}
```

为控制 README 长度，上述 JSON 只展示能够体现差异的原始条目；表格中的关键词列表来自
两份完整输出经过相同后处理后的结果。

</details>

## 评测结果

以下结果来自固定开发集上的离线评测。**宏平均 F1（Macro-F1）**表示对各样本 F1
等权平均，减少长关键词列表对总分的主导；**Schema 合法率**表示输出满足规定字段、
类型和结构的比例。

| 模型阶段 | Macro-F1 | Schema 合法率 |
|---|---:|---:|
| 基座模型 | 0.2490 | 0.4560 |
| SFT | 0.4946 | 0.6737 |
| DPO | 0.5437 | 0.7432 |
| GRPO | 0.5759 | 0.7976 |
| GRPO + 确定性合约修复 | 0.6932 | 1.0000 |

在另一个包含 496 条人工复核样本的隔离测试集上，GRPO 相对 DPO 的 Macro-F1、
Micro-F1 和 Schema 合法率分别提升 `0.0541`、`0.0701` 和 `0.0786`。该测试集未参与
训练或模型选择。

为了验证 Schema 可扩展性，项目还使用 240 条人工复核的合成客服工单测试了固定字段
路由任务。在其中 100 条隔离测试样本上，普通解码与严格 JSON Schema 解码得到相同的
质量指标：

| 指标 | 普通解码 | 严格 JSON Schema |
|---|---:|---:|
| Intent Macro-F1 | 0.8508 | 0.8508 |
| 紧急程度准确率 | 0.6500 | 0.6500 |
| Schema 合法率 | 1.0000 | 1.0000 |
| 完整语义合约合法率 | 0.9900 | 0.9900 |
| P50 延迟 | 0.175 s | 0.227 s |
| 吞吐 | 497.2 tok/s | 368.2 tok/s |

在这个任务上，严格约束解码没有提高质量，但 P50 延迟增加约 `29.7%`，吞吐下降约
`26.0%`。约束解码是否启用，应依据实际错误分布和性能预算决定。

## 项目结构

```text
core/        Schema、配置、推理、预处理、后处理与语义合约
prompts/     关键词任务 Prompt 构建
rl/          SFT、DPO、GRPO、LoRA 合并与 Reward 实现
examples/    关键词抽取、工单路由和少量数据格式示例
scripts/     CPU quickstart 与 vLLM 启动脚本
tests/       核心 Schema、推理与 Reward 测试
```

## 适用边界

- 当前质量结果主要来自中文电商评论关键词抽取。
- 客服工单路由数据是小规模模板合成数据，不代表真实客服流量表现。
- 确定性修复只能处理规则明确的错误，不能替代人工复核。
- 离线指标和单机服务基准不等同于生产环境 SLO。
- GitHub 仓库不直接包含模型权重、完整训练数据、内部评测报告或生产部署配置；公开合并
  权重单独托管在上述 Hugging Face 模型页。

## License

[MIT License](LICENSE)
