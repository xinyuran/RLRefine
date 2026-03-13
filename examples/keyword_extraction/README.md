# 关键词提取示例

本示例展示如何使用 StructAlign 框架进行关键词提取。

## 运行方法

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

- `schema.py`: 定义关键词提取的 JSON Schema
- `config.py`: 任务配置
- `run.py`: 运行脚本
- `sample_data.jsonl`: 示例数据

## 自定义任务

参考主 README.md 的 [自定义任务](#自定义任务) 章节。
