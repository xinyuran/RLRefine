[English](README_EN.md) | [中文](README_CN.md)

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

详细文档请查看 [中文版 README](README_CN.md)。

## 文档索引

| 文档 | 内容 |
|------|------|
| [README_CN.md](README_CN.md) | 完整中文文档（安装、API、训练流程） |
| [README_EN.md](README_EN.md) | English documentation |
| [docs/reward_design.md](docs/reward_design.md) | 奖励函数设计详解 |
| [docs/training_details.md](docs/training_details.md) | 训练细节与工程记录 |
| [docs/evaluation.md](docs/evaluation.md) | 评测方案与结果 |
| [examples/sample_data/](examples/sample_data/) | 训练数据样本 |

## License

MIT
