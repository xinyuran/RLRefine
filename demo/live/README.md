---
title: RLRefine Live Model Demo
emoji: 🔎
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
models:
  - xinyuran/Qwen2.5-7B-RLRefine
datasets:
  - xinyuran/RLRefine-Dataset
---

# RLRefine live model demo

Enter a Chinese review and click **Extract keywords**. The published
`xinyuran/Qwen2.5-7B-RLRefine` model generates the output on Hugging Face ZeroGPU.
The application displays keywords, source matches, contract errors, JSON and the
unmodified raw model response. No user-supplied JSON is needed.

Model revision: `bd15e8e02859b1c31d667499ee581783ccd79728`.
The app reuses `prompts/prompt_template_3.py` and `core/target_contract.py`.
It converts only finite numeric score strings (for example, `"0.95"`) to numbers,
explicitly labels this normalization and validates the resulting JSON. Keywords
and explanations are unchanged, and the unmodified raw response remains visible.
It does not change weights, quantize, retry generation or substitute rule-based
extraction. Remaining contract errors are displayed.

## Run or deploy

Install `demo/live/requirements.txt` in a compatible CUDA environment and run
`python demo/live/app.py` from the repository root. The ordinary repository
dependencies are also needed when running inside the full checkout.

For a Gradio Space, use ZeroGPU hardware and place these files at its root:

- `demo/live/app.py` → `app.py`
- `demo/live/inference.py` → `inference.py`
- `demo/live/requirements.txt` → `requirements.txt`
- This README → `README.md`
- `core/target_contract.py`, `core/keyword_contract.py` (keep their paths)
- `prompts/prompt_template_3.py` (keep its path)
- `LICENSE`

The Space uses these two directories as Python namespace packages; the rest of
the framework is not required for this small deployment. Model weights download
from the pinned public Hugging Face revision on startup.

## Limits

- Input: 1–512 characters; generation: up to 2,048 tokens.
- One inference at a time, with up to 8 queued requests.
- ZeroGPU queues and per-visitor daily quotas apply. The first visit may wait
  for a sleeping Space to start.
- Reviews are sent to this Space for inference. The application does not save
  inputs, outputs or user analytics.
- Contract validity is not task accuracy. Scores are model-generated values,
  not calibrated probabilities. This deployment is not a new quality benchmark.

[Source](https://github.com/xinyuran/RLRefine) ·
[Developer contract validator](https://huggingface.co/spaces/xinyuran/RLRefine-Playground)
