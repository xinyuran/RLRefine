# Using RLRefine

[English overview](../README.md) · [中文说明](../README_CN.md)

## Choose a starting point

| Goal | Entry point | Requires a model? |
|---|---|---|
| Explore JSON and source constraints | [Contract playground](../demo/README.md) | No |
| Check the Python installation | `python -m scripts.contract_quickstart` | No |
| Extract keywords from reviews | `python examples/keyword_extraction/run.py` | Yes |
| Define a fixed-field task | [Ticket-routing schema](../examples/intent_routing/README.md) | Only for generation |

## Install the core

```bash
git clone https://github.com/xinyuran/RLRefine.git
cd RLRefine
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m scripts.contract_quickstart
```

On Windows PowerShell, replace the activation command with
`.venv\Scripts\Activate.ps1`. The quickstart validates two hand-written payloads;
it does not generate keywords or measure model quality.

## Run the published model

The optional training/serving stack requires a compatible Linux/CUDA environment.
Installing it and starting the model may download large packages and weights.
If you already have an OpenAI-compatible service, skip the server installation
and configure the client for that service instead.

Server terminal:

```bash
python -m pip install -r requirements-training.txt
bash scripts/run_vllm.sh 0 8001 xinyuran/Qwen2.5-7B-RLRefine
```

Client terminal, from the repository root:

```bash
cp examples/keyword_extraction/.env.example examples/keyword_extraction/.env
python examples/keyword_extraction/run.py
```

In PowerShell use `Copy-Item examples/keyword_extraction/.env.example
examples/keyword_extraction/.env` for the copy step. The copied configuration is:

```dotenv
VLLM_BASE_URL=http://localhost:8001/v1
VLLM_API_KEY=dummy
MODEL_NAME=xinyuran/Qwen2.5-7B-RLRefine
DEBUG=false
```

For a remote service, replace the endpoint, model name and API key with values
for that service. A local model directory may be passed to `run_vllm.sh`; the
client's `MODEL_NAME` must then match the model name exposed by that server.
Do not commit a populated `.env` file. GPU capacity and latency depend on the
backend, context length and concurrency; this guide is not a serving SLA.

## Inspect results

`RLRefineProcessor` returns status and data fields. Inspect `status`, `fallback`,
`error_code` and `validation_errors` when present; do not count a fallback as a
successful model generation. The processor performs Schema checks and keyword
post-processing. For an explicit full keyword-contract check on the returned
payload, call the public validator with the corresponding source text:

```python
from core.target_contract import validate_keyword_payload

source = "包装很好，屏幕清晰。"
payload = {"keywords": [["原文对象", "屏幕", 0.94]]}
valid, errors = validate_keyword_payload(payload, source)
assert valid, errors
```

The keyword contract requires exactly one top-level `keywords` field, 1–15
three-item arrays, non-empty explanations, 1–4-character keywords, numeric
confidence in [0, 1], no duplicate keywords, and exact source substrings.
It does not establish explanation correctness, keyword relevance/completeness,
confidence calibration, or suitability for a new domain.

The routing example provides a task schema rather than a complete hosted
application. Its evidence field needs a separate source-membership check in
an application's generation path.
