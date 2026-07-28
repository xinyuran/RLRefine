# 标注与仲裁指南

本指南合并关键词冻结测试集与客服工单第二 Schema 的标注规则。它是操作规范，不是模型
选择文档。

## 1. 通用原则

- Source、sample ID、group ID、split 和 provenance 不可修改。
- 标注员不得看到模型身份，也不得根据模型分数选择标签。
- 所有标签必须得到原文支持。
- 排除样本必须提供明确原因。
- Dev 可用于 prompt/Schema 工程；冻结 test 标签只允许一次最终报告。
- Challenge 只做诊断。

## 2. 关键词 gold test

状态：496 条数据已冻结为 `keyword-gold-v1`。它们只用于最终 one-shot 报告，禁止用于
训练、prompt 选择、checkpoint 选择或阈值调整。

### 角色

- 主标注员：独立完成第一份标签。
- 副标注员：对同一冻结 source 独立标注。
- 仲裁员：只处理分歧，且必须与两位标注员不同。

### 关键词规则

- 除明确排除的样本外，输出 1–15 个关键词。
- 每个关键词必须是原文中的 1–4 字连续子串。
- 关键词不能重复。
- 优先选择原子概念，不合并多个短语。
- 不推断原文中未出现的情感、实体或原因。
- Confidence 必须为 `[0,1]` 数值，且不能替代证据。

### 排除规则

只有 source 为空/损坏、没有有意义的可抽取内容或超出任务边界时才能排除。排除行不含
关键词，并必须填写原因。

### 仲裁

仲裁员接收不可修改的 source 列和两份独立标签。对每个分歧，可选择主标、副标，或给出
受原文支持的仲裁标签及简短原因。不得编辑不可变列，也不得不经规则判断直接复制模型
输出。

冻结前必须：

1. 校验 sample ID 与不可变列；
2. 校验关键词长度、唯一性、source 成员关系、数量和 confidence；
3. 确认仲裁员独立；
4. 确认与 train/dev/challenge 零重叠；
5. 冻结行数、数据版本与 SHA256。

## 3. 客服工单第二 Schema

目的：证明框架可以从可变长关键词列表适配到固定对象。这是 smoke test，不是生产泛化
实验。

### 输出合约

| 字段 | 允许值 |
|---|---|
| `intent` | `refund`、`delivery`、`product_quality`、`account`、`other` |
| `urgency` | `low`、`normal`、`high` |
| `evidence` | 从工单原文复制的 1–40 字连续子串 |

### 数据集

- 共 240 条：120 dev、100 冻结 test、20 challenge。
- 当前数据由确定性模板合成，不包含外部文本或关键词任务文本；manifest、README 和报告
  均明确标记为 synthetic。
- 冻结前每一行都必须人工复核；合成数据不能直接视为 human gold。
- 重复模板和近重复样本按 group 管理后再切分。
- Intent 类别尽量平衡；不可避免的不平衡必须记录，不能静默重采样冻结 test。

### 标注规则

- `intent` 选择用户最主要的处理诉求。
- 只有明确时间敏感、安全、资金锁定或多次未解决时才标为 `urgency=high`；不能只根据
  情绪词推断。
- `evidence` 选择能够直接支持 intent 与 urgency 判断的最短原文片段。
- 没有已声明类别适用时使用 `other`，评测期间不得扩展 enum。

### Split 用途

- Dev：固定一个 prompt，并验证 Schema/runner。
- 冻结 test：只做一次 unconstrained-vs-constrained 最终比较。
- Challenge：诊断多意图、否定、urgency 歧义、prompt injection 和长上下文。

冻结 test 不得触发 prompt sweep、训练、DPO/GRPO、阈值修改或新的 serving matrix。

### 冻结结果

复核于 2026-07-28 完成：240 条全部批准，不需要修改标签；数据冻结在
`data/frozen/intent_routing_v1/`。Manifest 绑定 review packet、split 哈希、复核人、
日期、行数和禁止用途。冻结 test 已完成一次预注册比较，之后不得再用于选择或调参。

## 4. 审计清单

- [x] 已记录 provenance 与 synthetic 披露；
- [x] 不可变列未修改；
- [x] 跨 split 的 sample/group overlap 为零；
- [x] 标签 enum 与 evidence-source 成员关系合法；
- [x] 冻结 test 哈希和行数已记录；
- [x] run manifest 已记录 human-gold 使用边界。
