# 关键词抽取示例

本示例展示如何使用 StructAlign 对中文电商评论执行 Schema 约束关键词抽取。

> **说明：** 该任务由作者 rxy 针对中文文本设计和验证，英文效果尚未充分评估。

## 运行方式

### 1. 启动 vLLM 服务

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

### 2. 运行示例

```bash
cd StructAlign
python examples/keyword_extraction/run.py
```

## 文件说明

- `schema.py`：定义关键词抽取 JSON Schema；
- `config.py`：任务配置；
- `run.py`：示例入口；
- `sample_data.jsonl`：示例数据。

主实验协议、最终指标和限制见
[`docs/experiment_report.md`](../../docs/experiment_report.md)。如需定义其他任务，可参考
[`examples/intent_routing/`](../intent_routing/) 的固定对象 Schema。
