# 关键词抽取示例

本示例展示如何使用 RLRefine 从中文电商评论中抽取带说明和置信度的关键词。

## 输出结构

```json
{
  "keywords": [
    ["原文评价包装", "包装", 0.96],
    ["原文描述屏幕", "屏幕", 0.94]
  ]
}
```

每个关键词必须是原文中的连续子串，长度、数量、重复项和置信度类型都会经过校验。

## 运行方式

启动 OpenAI-compatible 模型服务：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

配置并运行示例：

```bash
cp examples/keyword_extraction/.env.example examples/keyword_extraction/.env
python examples/keyword_extraction/run.py
```

## 文件说明

- `schema.py`：关键词字段与约束；
- `config.py`：推理配置；
- `run.py`：示例入口；
- `sample_data.jsonl`：少量输入样例。

固定字段分类任务可参考 [`examples/intent_routing/`](../intent_routing/)。
