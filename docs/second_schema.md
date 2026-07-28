# 第二 Schema：客服工单路由

## 当前状态

第二 Schema smoke test 已完成。合约、预注册、240 条人工复核合成数据、冻结 split 哈希、
预测、benchmark 和独立结果审计均已保留。

240 条数据于 2026-07-28 全部批准，不需要修改标签。冻结 manifest 位于
`data/frozen/intent_routing_v1/manifest.json`，最终 benchmark 位于
`reports/second_schema/serving/benchmark.json`。

预注册文件为 `reports/second_schema/preregistration.json`。实验未使用关键词数据或
关键词专用 E4 adapter。

## 合约与范围

第二 Schema 是固定对象的分类加证据任务，与关键词列表的输出结构不同。

| 字段 | 约束 |
|---|---|
| `intent` | `refund`、`delivery`、`product_quality`、`account`、`other` |
| `urgency` | `low`、`normal`、`high` |
| `evidence` | 从 source 工单复制的 1–40 字连续子串 |

目的是证明框架能改变输出结构和语义校验逻辑，不是 transfer learning、SOTA、真实客户
泛化或部署实验。

- 数据：240 条确定性合成工单，120 dev、100 test、20 challenge；不复用外部文本或
  关键词任务文本。
- 实验臂：同一冻结 Base 模型与 prompt，分别运行 unconstrained 和 strict JSON-Schema
  decoding；不加载 E4。
- 算力：1 张 GPU，固定 concurrency `4`，不训练。
- 指标：Intent Macro-F1、urgency accuracy、evidence exact/来源忠实率、形式 Schema
  合法率、完整 target-contract 合法率、请求失败、墙钟延迟和吞吐。
- 决策规则：constrained Schema `>=0.99`、target-contract `>=0.95`，Intent Macro-F1
  不得比 unconstrained 低超过 `0.03`；challenge 只做诊断。
- Stop rule：无论结果正负，只报告这一次比较，不追加 prompt sweep、DPO/GRPO、量化或
  serving matrix。

## 最终结果

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

决策：`SECOND_SCHEMA_SMOKE_TEST_PASS`；`deployment_authorized=false`。

Strict decoding 的冻结 test 输出完全相同，却使 P50 延迟增加 `29.7%`、吞吐下降
`26.0%`。因此，对这个有限的合成任务，观测到的服务选择是 Base unconstrained 加语义
校验/quarantine。这不代表 unconstrained decoding 在更广泛真实流量上仍能保持结构可靠。

## 复现数据冻结：不需要 GPU

已复核的 UTF-8 packet：

```text
data/derived/intent_routing_v1_draft/review_packet.csv
```

所有行均已包含 `approved=true`。重新生成冻结文件：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m scripts.freeze_second_schema_dataset \
  --reviewer YOUR_NAME \
  --review-date YYYY-MM-DD
python -m unittest -v \
  tests.test_intent_routing_schema \
  tests.test_second_schema_assets \
  tests.test_second_schema_benchmark
```

如果存在未批准、标签非法、evidence 不属于 source，或不可变 draft 字段/hash 改变，
freeze 命令会失败。

## 复现单卡比较

在同一服务器使用两个终端。只有 vLLM 进程需要 GPU。固定 float16 Qwen2.5-7B runner
使用 1 张 GPU，实践中建议 24 GB；客户端除调用服务外只使用 CPU。

终端 A：

```bash
source .venv/bin/activate
python -m pip install -r requirements-training.txt
bash scripts/run_second_schema_vllm.sh 0 8003 /path/to/Qwen2.5-7B-Instruct
```

该命令在示例物理 `GPU 0` 上启动指定的 Base snapshot。等待服务 ready；不得加载 E4 或
其他 adapter。

终端 B：

```bash
source .venv/bin/activate
bash scripts/run_second_schema_smoke_test.sh \
  Qwen/Qwen2.5-7B-Instruct \
  http://127.0.0.1:8003
```

客户端固定 concurrency 4，不额外占用 GPU；它对 dev/test/challenge 各运行两个实验臂，
并写入 `reports/second_schema/serving/`。复现结果不得用于重新调参或解除 stop rule。
