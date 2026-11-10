# Contributing to RLRefine

Useful contributions include reproducible extraction failures, clearer examples,
evaluation improvements, and independently evaluated new task schemas.

## Set up and check a change

Use Python 3.10+ and install `requirements.txt`. The public checks do not need a
GPU, model download, API key or external inference service:

```bash
python -m scripts.contract_quickstart
python -m unittest discover -v tests
```

Node.js 22+ additionally runs the browser/Python validator parity test. Without
Node, that test is reported as skipped. Preview the static demo with:

```bash
python -m http.server 7860 --bind 127.0.0.1 --directory demo
```

## Report a failure

Open an [issue](https://github.com/xinyuran/RLRefine/issues/new) with the smallest
non-sensitive input, expected behavior, actual output, commit, Python version,
and model/backend versions if inference is involved. Remove API keys and private
customer text. A hand-written reproduction is useful when real text cannot be shared.

## Send a pull request

Keep the change focused and explain the behavior it changes. Add a regression
test for new logic or a bug fix, and include the checks you actually ran.
If the keyword contract changes, update the Python rules, browser equivalent,
parity tests, and user-facing documentation together.

Changes to evaluation must distinguish raw model output, post-processing,
fallbacks and errors. Report dataset provenance and split size; keep development
results separate from held-out evaluation. Do not present contract validity as
task accuracy or a small local benchmark as production performance.

The project uses the [MIT License](LICENSE).
