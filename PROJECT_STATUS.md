# StructAlign 项目状态

> 最后更新：2026-07-28
> 本文件是项目当前状态的唯一事实来源。

## 最终结论

计划内的模型实验已经全部结束。

- 关键词主线 BP1–BP6：
  `RETAIN_REPAIRED_E4_FOR_OFFLINE_REVIEW`，`deployment_authorized=false`。
- 第二 Schema 客服工单 smoke test：
  `SECOND_SCHEMA_SMOKE_TEST_PASS`，`deployment_authorized=false`。
- 预注册 stop rule 已生效，不再进行训练、prompt sweep、decoding matrix、量化实验或
  模型选择。

## 关键词主线证据

| 阶段 | Dev Macro-F1 | Dev Micro-F1 | Dev Schema |
|---|---:|---:|---:|
| Base v2 | 0.2490 | — | 0.4560 |
| SFT | 0.4946 | — | 0.6737 |
| DPO | 0.5437 | 0.5702 | 0.7432 |
| E4 | 0.5759 | 0.5975 | 0.7976 |
| E4 + BP6 修复 | 0.6932 | 0.6774 | 1.0000 |

在 496 条冻结 human-gold one-shot 上，E4 的 Macro/Micro-F1/Schema 为
`0.352544/0.444791/0.552419`，相对 DPO 提升
`+0.054133/+0.070073/+0.078629`。

保留的离线策略为：

```text
冻结 E4
  -> strict JSON-Schema 解码
  -> 确定性 target-contract 校验/修复
  -> 不可恢复样本进入 quarantine
  -> 人工复核
```

## 第二 Schema 证据

独立的客服工单任务使用 240 条确定性合成数据，冻结前全部完成人工复核：
120 条 dev、100 条 one-shot test、20 条诊断 challenge。实验只使用 Base 模型、
concurrency 4，不训练，也不加载 E4 adapter。

| 冻结 test 指标 | Unconstrained | Strict JSON Schema |
|---|---:|---:|
| Intent Macro-F1 | 0.850810 | 0.850810 |
| Urgency accuracy | 0.650000 | 0.650000 |
| Evidence exact | 0.390000 | 0.390000 |
| 来源忠实率 | 0.990000 | 0.990000 |
| Schema 合法率 | 1.000000 | 1.000000 |
| Target-contract 合法率 | 0.990000 | 0.990000 |
| P50 延迟 | 0.175011 s | 0.226956 s |
| 吞吐 | 497.232 tok/s | 368.190 tok/s |

所有预注册检查均通过。Strict decoding 没有冻结 test 质量收益，却使 P50 延迟增加
`29.7%`、吞吐下降 `26.0%`。因此，该小规模合成任务的观测策略是 Base
unconstrained 加语义校验/quarantine；这不覆盖关键词任务的独立策略，也不能证明真实
客服流量上的表现。

归档、预注册、数据集、预测和报告哈希全部匹配；仓库 evaluator 已逐字段复算并确认所有
指标一致。审计证据位于：

- `reports/second_schema/result_audit.json`；
- `reports/second_schema/serving/`；
- `data/frozen/intent_routing_v1/`。

## 发布状态

模型实验已冻结。公开发布只包含源代码、维护文档、合成冻结数据和经过清理的第二
Schema 审计结果；模型权重、原始/派生训练数据、日志、个人路径和面试材料不发布。
基础依赖与 GPU 训练/服务依赖已拆分，并提供 CPU-only 合约 quickstart。

## 维护文档

- [`README.md`](README.md)：中文公共入口；
- [`docs/experiment_report.md`](docs/experiment_report.md)：完整实验链路；
- [`docs/annotation_guide.md`](docs/annotation_guide.md)：标注规则；
- [`docs/second_schema.md`](docs/second_schema.md)：第二 Schema 协议与结果；
- [`LICENSE`](LICENSE)：MIT License。
