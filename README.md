# StructAlign

一个面向 Schema 约束信息抽取的、评测驱动的大模型后训练与服务实践项目。

项目以中文电商评论关键词抽取为主任务，将版本化数据、冻结评测边界、
SFT/DPO/GRPO、Reward 验证、约束解码、语义合约修复和部署门控串成一条可复现的实验
链路。目标不是搭建完整平台，而是保留可信实验、真实工程问题和可供面试深挖的决策
过程。

## 已验证结果

关键词主线已经完成至 BP6。

| 阶段 | Dev Macro-F1 | Dev Schema 合法率 |
|---|---:|---:|
| Base v2 | 0.2490 | 0.4560 |
| SFT | 0.4946 | 0.6737 |
| DPO | 0.5437 | 0.7432 |
| E4 GRPO-from-DPO | 0.5759 | 0.7976 |
| E4 + 确定性修复 | 0.6932 | 1.0000 |

在 496 条冻结 human-gold one-shot 上，E4 相对 DPO 的 Macro/Micro-F1/Schema 提升为
`+0.0541/+0.0701/+0.0786`。

最终关键词任务策略保持保守：

```text
冻结 E4
  -> strict JSON-Schema 解码
  -> 确定性语义合约校验/修复
  -> 不可恢复样本进入 quarantine
  -> 人工复核
```

决策为 `RETAIN_REPAIRED_E4_FOR_OFFLINE_REVIEW`，不授权自动线上部署
（`deployment_authorized=false`）。

## 第二 Schema 结果

Support-ticket routing smoke test 已在 240 条人工复核的合成数据上完成。在 100 条冻结
test 上，Base unconstrained 与 strict JSON Schema 输出完全相同：

| 指标 | Unconstrained | Strict JSON Schema |
|---|---:|---:|
| Intent Macro-F1 | 0.850810 | 0.850810 |
| Urgency accuracy | 0.650000 | 0.650000 |
| Schema 合法率 | 1.000000 | 1.000000 |
| 语义合约合法率 | 0.990000 | 0.990000 |
| P50 延迟 | 0.175011 s | 0.226956 s |
| 吞吐 | 497.232 tok/s | 368.190 tok/s |

Strict decoding 没有带来测试质量收益，但 P50 延迟增加 `29.7%`、吞吐下降 `26.0%`。
该结果只证明框架可以适配不同 Schema，不代表真实客服业务泛化或部署授权。预注册
stop rule 已执行，模型实验到此结束。

## 项目体现的能力

- 数据血缘、group-aware split、challenge 诊断和冻结 human-gold 隔离；
- Base、SFT、DPO、GRPO 使用同一评测协议；
- Reward 组件测试、攻击样本和失败安全的阶段授权；
- ms-swift、LoRA adapter、插件加载和 vLLM 的真实集成排障；
- 区分形式 JSON 合法、语义业务合约合法和 transport 成功；
- 用批次墙钟计算并发吞吐，并根据质量收益与性能成本选择服务策略；
- 保留负向证据，不为了展示而夸大部署结论。

## 工程验证

基础安装要求 Python 3.10+，不包含 `ms-swift`、`vLLM` 或模型权重，可在普通 CPU
环境完成：

```bash
git clone https://github.com/xinyuran/RLRefine.git
cd RLRefine
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m scripts.contract_quickstart
python -m unittest discover -v tests
```

Windows PowerShell 只需将激活命令替换为 `.venv\Scripts\Activate.ps1`。

CPU quickstart 只验证两种 Schema 与语义合约，不下载模型、不调用外部 API，也不运行
实验。GPU 训练/服务依赖单独位于 `requirements-training.txt`；相关 runner 位于
`scripts/`，并受阶段授权或冻结清单控制。历史结果应从冻结配置和报告复现，不应盲目
重跑所有训练流程。

## 文档入口

- [`PROJECT_STATUS.md`](PROJECT_STATUS.md)：当前唯一项目状态；
- [`docs/experiment_report.md`](docs/experiment_report.md)：完整实验链路与结果；
- [`docs/annotation_guide.md`](docs/annotation_guide.md)：标注与仲裁规范；
- [`docs/second_schema.md`](docs/second_schema.md)：第二 Schema 协议、结果和复现命令；
- `core/`、`evaluation/`、`rl/`、`scripts/`、`configs/`：实现与可复现入口。

## 局限

- 主任务仅覆盖中文电商关键词抽取。
- E4 是研究候选，不是线上生产模型。
- 第二 Schema 使用小规模、模板合成且人工复核的数据，不代表真实业务泛化。
- 历史 BP5 throughput 不是墙钟并发吞吐，不能作为生产 SLO。
- 两条第二 Schema 合约失败把 prompt 包装文本 `工单：` 复制进了 evidence，说明 strict
  JSON Schema 仍不能保证来源忠实性。

代码采用 MIT License。公开的 `data/frozen/intent_routing_v1/` 是本项目确定性生成并
人工复核的合成数据；原始/派生训练数据、模型权重、运行日志、个人材料和本地路径均不
包含在发布包中。
