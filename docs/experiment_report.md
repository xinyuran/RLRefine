# StructAlign 实验报告

> 状态：关键词抽取 BP1–BP6 与第二 Schema smoke test 均已完成。
> 所有最终决策保持 `deployment_authorized=false`，模型实验 stop rule 已生效。

## 1. 项目问题

在不把最终人工测试集泄漏给模型选择的前提下，一个 Schema 约束信息抽取项目能否通过
数据治理、SFT、偏好优化、Reward 优化、约束解码和确定性校验获得可信收益？

主任务是中文电商评论关键词抽取。输出合约为包含 1–15 个关键词 tuple 的 JSON 对象；
关键词必须为原文中的 1–4 字连续子串，且不能重复。

## 2. 评测边界

- 训练和模型选择只使用版本化 train 与 teacher-dev。
- Challenge 只做诊断，不参与模型选择。
- 496 条冻结 human-gold 仅用于一次最终报告，不参与 prompt、checkpoint、阈值或模型选择。
- 每个阶段在同一 evaluator 下与直接上游比较。
- 后续人工研究决策可以继续实验，但不能改写历史机械 gate。
- 形式 JSON-Schema 合法率与完整语义 target-contract 合法率分开报告。

## 3. 实验链路

| 阶段 | Dev Macro-F1 | Dev Micro-F1 | Dev Schema | 解释 |
|---|---:|---:|---:|---|
| Base v2 | 0.2490 | — | 0.4560 | 冻结协议下的公平基线 |
| SFT | 0.4946 | — | 0.6737 | 任务收益明显，生产 gate 拒绝 |
| DPO | 0.5437 | 0.5702 | 0.7432 | 相对 SFT 可靠提升，部署拒绝 |
| E4 GRPO-from-DPO | 0.5759 | 0.5975 | 0.7976 | 最优研究候选，机械 gate 拒绝 |
| E4 + BP6 修复 | 0.6932 | 0.6774 | 1.0000 | 仅保留用于离线人工复核 |

### Base 与 SFT

Base v2 在新 target contract 下建立公平基线，Dev Macro-F1 为 `0.2490`，Schema 合法率为
`0.4560`。SFT 将 Dev Macro-F1 提升到 `0.494647`，配对增益为 `+0.245615`，95% CI
为 `[0.201626, 0.288973]`；Schema 合法率提升到 `0.673716`。

但输出长度、幻觉和截断仍未达到生产 gate。由此明确：研究晋级与部署晋级必须使用不同
门槛；模型可以学到任务，但仍不适合直接服务。

### DPO

DPO checkpoint-74 的 Dev Macro/Micro-F1 为 `0.543675/0.570223`，Schema 为
`0.743202`。相对 SFT 的配对 Macro-F1 增益为 `+0.049028`，95% CI 为
`[0.014690, 0.084062]`；Challenge Macro-F1/Schema 为 `0.357590/0.535714`。

由于绝对 Schema 和输出长度阈值未命中，冻结机械 gate 保持 REJECT。项目通过透明记录的
人工研究决策只授权一次 GRPO-from-DPO，同时保留原始拒绝证据。

### E4 GRPO-from-DPO

E4 完成 295 steps，并冻结最终 checkpoint。相对 DPO，Dev Macro/Micro-F1 分别提升
`+0.032185/+0.027317`，Schema 提升 `+0.054381`。Macro-F1 95% CI
`[-0.003390, 0.068316]` 跨过零；Schema `0.797583` 比预注册 `0.80` 少一个合法样本，
因此研究与部署机械 gate 均保持 REJECT。

E4 仍被冻结为最优研究候选，因为它在核心质量、Schema、幻觉、长度、延迟和真实截断
方向都优于 DPO，且选择过程中未使用 challenge 或 human gold。

### 冻结 human-gold one-shot

| 模型 | Macro-F1 | Micro-F1 | Schema |
|---|---:|---:|---:|
| DPO | 0.298411 | 0.374718 | 0.473790 |
| E4 | 0.352544 | 0.444791 | 0.552419 |
| E4 - DPO | +0.054133 | +0.070073 | +0.078629 |

这 496 条数据只用于报告泛化结果，没有重新开启模型选择或调参。

## 4. 错误分类

项目直接分类已有 E4 失败，不重新推理。

| Split | 失败数 | 主要类别 |
|---|---:|---|
| Dev | 67/331 | 长度/类型 31；非原文 23；重复 11；tuple 形状 2 |
| Challenge | 49/140 | 非原文 19；长度/类型 14；重复 10；数量 4；JSON 1；tuple 形状 1 |

剩余问题主要是语义合约执行，而不是 completion 缺失或 token 截断。

## 5. 服务与约束解码

vLLM benchmark 在同一 E4 上比较 unconstrained 与 strict JSON-Schema 解码。

| Concurrency | Strict Schema | Unconstrained Schema | Strict target-contract 失败 |
|---:|---:|---:|---:|
| 1 | 0.851964 | 0.782477 | 49 |
| 4 | 0.839879 | 0.779456 | 53 |
| 8 | 0.885196 | 0.785499 | 38 |

所有 completion 均正常结束，没有达到最大 token。历史字段 `runtime_error_count` 实际统计
的是 target-contract 失败，不是 transport 或 vLLM 崩溃；历史 throughput 使用逐请求耗时
之和，而不是批次墙钟，因此不作为生产 SLO。

Strict JSON Schema 改善了形式结构，但无法保证来源忠实性或去重。BP5 只得到离线 batch
候选结论，从未获得在线部署授权。后续 telemetry v2 已拆分 transport error 与
target-contract error，并改用批次墙钟吞吐；冻结历史报告不改写。

## 6. BP6 确定性合约修复

BP6 不训练、不重新生成、不修改 prompt/decoding，也不使用 human gold。它只保留满足冻结
target contract 的已有 tuple，不发明或改写关键词；无法恢复的输出继续保持 error。

| Split | 原始 Macro/Micro/Schema | 修复后 Macro/Micro/Schema | 已修复 | 不可恢复 |
|---|---|---|---:|---:|
| Dev | 0.575860 / 0.597540 / 0.797583 | 0.693238 / 0.677433 / 1.000000 | 67 | 0 |
| Challenge | 0.427209 / 0.453564 / 0.650000 | 0.619733 / 0.592354 / 0.992857 | 49 | 1 |

精确覆盖、未使用 human gold、模型/prompt/decoding 不变、修复输出合法、Macro-F1
non-inferiority 等所有预注册检查均通过。

最终决策：

```text
RETAIN_REPAIRED_E4_FOR_OFFLINE_REVIEW
deployment_authorized=false
```

保留的关键词任务模式为：

```text
冻结 E4
  -> strict JSON-Schema 解码
  -> 确定性语义合约校验/修复
  -> 不可恢复样本进入 quarantine
  -> 人工复核
```

## 7. 工程经验

项目刻意保留失败启动和错误解释，作为工程证据。主要面试案例包括：

- 外部 Reward plugin 必须从真实 import 边界测试，而不只是仓库内单测；
- LoRA 延续训练必须区分 Base、policy adapter 和 reference adapter；
- 第三方 batch 代数约束与 registry factory 语义必须成为显式配置合约；
- batched EOS/padding 不能直接作为单序列截断证据；
- 研究晋级 gate 与部署晋级 gate 必须分离；
- 形式 JSON Schema 不等于语义业务合约；
- 并发吞吐必须使用批次墙钟时间。

完整问题、解决方法和证据见 [`../EXPERIENCE.md`](../EXPERIENCE.md)。

## 8. 第二 Schema smoke test

最终实验把可变长关键词列表切换为固定客服工单对象，字段包括 intent、urgency 和原文
evidence。共使用 240 条确定性合成数据，冻结前全部完成人工复核：
120 dev、100 one-shot test、20 诊断 challenge。实验不使用关键词数据、E4 adapter、
训练或 prompt sweep。

| 冻结 test 指标 | Base unconstrained | Base strict JSON Schema |
|---|---:|---:|
| Intent Macro-F1 | 0.850810 | 0.850810 |
| Urgency accuracy | 0.650000 | 0.650000 |
| Evidence exact | 0.390000 | 0.390000 |
| Evidence 来源忠实率 | 0.990000 | 0.990000 |
| 形式 Schema 合法率 | 1.000000 | 1.000000 |
| 完整 target-contract 合法率 | 0.990000 | 0.990000 |
| P50 端到端延迟 | 0.175011 s | 0.226956 s |
| Output-token 吞吐 | 497.232 tok/s | 368.190 tok/s |

所有预注册检查通过，决策为 `SECOND_SCHEMA_SMOKE_TEST_PASS`。Test 与 challenge 两臂输出
完全相同，只有一条 dev prediction 不同。Strict decoding 没有冻结 test 质量收益，
P50 延迟增加 `29.7%`，吞吐下降 `26.0%`。

两臂均无 transport error 或 max-token 输出。一条 test 和一条 challenge 输出因为把
prompt 包装文本 `工单：` 复制进 `evidence` 而违反语义合约；strict JSON Schema 无法
表达“必须是 source 子串”。Evidence exact 远低于来源忠实率，是因为模型常选择更长但
仍来自原文的证据片段。

该结果只支持有限的框架适配结论。对这个合成路由任务，不默认选择 strict decoding；
观测到的低成本策略是 unconstrained Base 加语义校验/quarantine。关键词任务继续使用
独立的 E4 + strict decoding + 确定性修复 + 人工复核策略。

预注册 stop rule 已执行，不再增加模型、prompt、decoding 或 serving 实验。

审计证据位于：

- `reports/second_schema/serving/`；
- `data/frozen/intent_routing_v1/`；
- `reports/second_schema/result_audit.json`。

## 9. 文档归档

合并前的 41 份 Markdown 保存在：

```text
reports/document_archive/markdown_before_consolidation_20260728_paths.zip
SHA256 A9055B02888A4C811452E07211FBF5919761C9170DEB136683FB0EB25079EE39
```
