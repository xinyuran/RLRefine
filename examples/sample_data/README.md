# 脱敏训练数据样本

本目录包含 RLRefine 三阶段训练中使用的数据格式示例（已脱敏）。

## 文件说明

| 文件 | 数据量 | 说明 |
|------|--------|------|
| `sft_sample.jsonl` | 2 条 | SFT 阶段训练数据格式 |
| `dpo_sample.jsonl` | 2 条 | DPO 阶段偏好对数据格式 |
| `grpo_sample.jsonl` | 2 条 | GRPO 阶段 prompt 数据格式（无 assistant 回复） |

## 数据格式

### SFT 格式

标准 ChatML 多轮对话格式，包含完整的 system + user + assistant：

```json
{
  "messages": [
    {"role": "system", "content": "关键词提取规则与格式要求..."},
    {"role": "user", "content": "请分析以下评论并提取关键词：\n\n【待处理评论】\n..."},
    {"role": "assistant", "content": "思考\n...(推理过程)...\n\n{\"keywords\": [...]}"}
  ]
}
```

assistant 回复由 Qwen3-Max 生成，包含"思考"推理段落和 JSON 输出两部分。

### DPO 格式

在 SFT 格式基础上，增加 `response`（chosen）和 `rejected_response`（rejected）字段：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "response": "高质量的思考+输出（chosen）",
  "rejected_response": "低质量的思考+输出（rejected）"
}
```

- **chosen**：遵循规则的高质量输出（原子化关键词、正确格式、充分推理）
- **rejected**：违反规则的低质量输出（未拆分、超长关键词、缺少推理、格式错误）

### GRPO 格式

仅包含 system + user prompt，不含 assistant 回复（由模型自行生成）：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

GRPO 训练时模型对每个 prompt 生成多个 completion，由奖励函数评分后计算策略梯度。

## 完整数据集

完整训练数据位于 `rl/data/` 目录（未公开），规模：
- SFT: 3,329 条训练 + 369 条验证
- DPO: ~3,697 条偏好对
- GRPO: 与 SFT 同源 prompt
