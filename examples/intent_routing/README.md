# 客服工单路由 Schema

> **状态（2026-07-28）：** 240 条合成数据已完成人工复核和冻结，预注册的 Base
> unconstrained-vs-strict 比较已经结束。完整协议见
> [`docs/second_schema.md`](../../docs/second_schema.md)。

这是 StructAlign 的第二 Schema 示例。它与关键词列表不同：每个输出都是固定对象，
包含枚举 intent、枚举 urgency，以及从工单原文复制的短 evidence。

```json
{"intent":"delivery","urgency":"high","evidence":"包裹还没送到"}
```

实验只使用 Base 模型，不复用关键词 gold 或 E4 adapter。冻结 test 上，两臂 Intent
Macro-F1 均为 `0.850810`，Schema 合法率均为 `1.000000`，target-contract 合法率均为
`0.990000`。Strict decoding 没有提升测试质量，反而降低吞吐，因此只能支持“框架已适配
第二种合成 Schema”的有限结论。
