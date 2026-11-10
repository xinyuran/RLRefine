---
title: RLRefine Contract Playground
emoji: 🔎
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
models:
  - xinyuran/Qwen2.5-7B-RLRefine
datasets:
  - xinyuran/RLRefine-Dataset
---

# RLRefine Contract Playground

An interactive, browser-only companion to [RLRefine](https://github.com/xinyuran/RLRefine).
Edit a Chinese review and JSON output to inspect structure, source matches,
duplicate keywords and confidence types. No model runs in this demo.

To generate keywords directly from a review, use the
[live model demo](https://huggingface.co/spaces/xinyuran/RLRefine-Demo).

The first three examples are hand-written teaching examples. The historical
example is a partial base-model output preserved in the previous project README;
it is not a live generation or an evaluation dataset.

The browser validator mirrors `core.target_contract.validate_keyword_payload`.
The repository's parity tests compare it with the Python implementation, including
invalid tuples, extra fields, Unicode, duplicates and source mismatches.

## Run locally

From the repository root, run `python -m http.server 7860 --bind 127.0.0.1 --directory demo`,
then visit <http://127.0.0.1:7860>. There is no build step or JavaScript dependency.

## Host on Hugging Face

Create a **Static** Space and upload this directory's top-level files at the Space root,
together with the repository's `LICENSE`. Keep the YAML metadata above in the
Space's root README. No GPU, API secret or inference endpoint is required.

Do not add service credentials to this static application. The separate
[live model demo](live/README.md) runs inference on ZeroGPU.

## Scope

Passing the contract does not guarantee useful keyword selection, completeness,
calibrated confidence, or production readiness. Input remains in the page; the
application has no storage, telemetry or inference requests.
