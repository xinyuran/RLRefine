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
bash scripts/run_vllm.sh 0 8001 xinyuran/Qwen2.5-7B-RLRefine
```

配置并运行示例：

```bash
cp examples/keyword_extraction/.env.example examples/keyword_extraction/.env
python examples/keyword_extraction/run.py
```

环境文件默认指向 `http://localhost:8001/v1` 和
`xinyuran/Qwen2.5-7B-RLRefine`。如果使用其他模型或服务，先同步修改端口与模型名。
Linux/CUDA 依赖安装、PowerShell 命令和返回状态说明见 [使用指南](../../docs/usage.md)。

## 文件说明

- `schema.py`：关键词字段与约束；
- `config.py`：推理配置；
- `run.py`：示例入口；
- `sample_data.jsonl`：少量输入样例。

固定字段分类任务可参考 [`examples/intent_routing/`](../intent_routing/)。
