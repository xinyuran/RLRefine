# 客服工单路由 Schema

该示例展示如何将 StructAlign 用于固定字段分类任务。输出包含工单意图、紧急程度和一段
直接复制自原文的证据：

```json
{
  "intent": "delivery",
  "urgency": "high",
  "evidence": "包裹还没送到"
}
```

Schema 对 `intent` 和 `urgency` 使用枚举约束，对 `evidence` 使用长度约束。实际应用还
应在生成后验证 evidence 是否确实属于输入文本，从而区分形式合法与来源忠实。

```python
from examples.intent_routing.schema import create_intent_routing_schema

schema = create_intent_routing_schema()
valid, errors = schema.validate(
    {
        "intent": "delivery",
        "urgency": "high",
        "evidence": "包裹还没送到",
    }
)
```
