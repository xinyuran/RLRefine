# Post-training recipes

[Overview](../README.md) · [Evaluation and scope](evaluation.md)

RLRefine includes LoRA-based SFT, DPO and GRPO recipes for Chinese review keyword
extraction. The published model is
[xinyuran/Qwen2.5-7B-RLRefine](https://huggingface.co/xinyuran/Qwen2.5-7B-RLRefine),
based on Qwen2.5-7B-Instruct. The maintainer has confirmed that the historical
research name **E4** and the released **checkpoint-924** refer to the same weights;
they are different names, not two benchmark candidates.

| Stage | Purpose | LoRA rank / alpha | Learning rate | Epochs | Setting |
|---|---|---:|---:|---:|---|
| SFT | Task and output-format learning | 16 / 64 | `1e-5` | 5 | Maximum length 8192 |
| DPO | Preference alignment | 8 / 32 | `5e-7` | 1 | `beta=0.2` |
| GRPO | Reward-based task optimization | 8 / 16 | `5e-7` | 1 | `beta=0.01`, 4 generations per prompt |

The reported training environment used 2 × NVIDIA H100 80GB, BF16,
ms-swift 3.11.2, vLLM 0.13.0 and DeepSpeed ZeRO-2. These are the documented
experiment settings, not minimum requirements for every use of the toolkit.

## Scripts and data

- [SFT](../rl/sft_finetune.sh)
- [DPO](../rl/dpo_finetune.sh)
- [GRPO](../rl/grpo_finetune.sh)
- [LoRA merge](../rl/merge_lora.sh)
- [Reward implementation](../rl/reward_builder.py)
- [Versioned JSON-target reward](../rl/reward_builder_v2.py)
- [Public synthetic training dataset](https://huggingface.co/datasets/xinyuran/RLRefine-Dataset)
- [Small data-format examples](../examples/sample_data/README.md)

Read and adapt GPU IDs, model/adapter paths, dataset paths, output directories
and training targets before executing a script. The scripts are recipes with
example paths; they are not a one-command reproduction of every historical run.

The dataset has `sft`, `dpo` and `grpo` configurations. Its original synthetic
training targets are preserved, including historical reasoning/output formats.
The current `core.target_contract` specifies JSON-only output. Do not silently
assume that all historical data already satisfy the newer contract. Validate
the selected data and prompt before training.

## Reward and evaluation

Rewards use explicit structure, source and task-reference checks. A high reward
is not an independent quality metric. Freeze development and held-out sets,
keep the held-out set out of model/prompt selection, and measure task F1 and
contract failures separately. The public training data are not an independent
held-out benchmark.

Training new weights, changing a prompt, or changing a serving backend creates
a new evaluation condition. Record that condition instead of attaching the
historical scores to an unmeasured configuration.
