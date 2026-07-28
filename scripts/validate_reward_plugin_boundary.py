"""Validate the Reward V2 plugin under ms-swift's standalone import boundary."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def validate_standalone_import(plugin: Path) -> dict[str, object]:
    plugin = plugin.resolve()
    if not plugin.is_file():
        raise ValueError(f"Reward plugin not found: {plugin}")
    code = (
        "import importlib, inspect, json, sys; "
        f"sys.path.insert(0, {str(plugin.parent)!r}); "
        f"m=importlib.import_module({plugin.stem!r}); "
        "assert hasattr(m, 'schema_based_reward_v2'); "
        "assert inspect.isclass(m.schema_based_reward_v2); "
        "assert m.schema_based_reward_v2 is m.SchemaBasedRewardV2; "
        "assert not m.HAS_SWIFT or m.orms.get('schema_based_reward_v2') is m.schema_based_reward_v2; "
        "print(json.dumps({'module': m.__name__, "
        "'registered_object': 'schema_based_reward_v2'}))"
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory() as temporary:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=temporary,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
    if result.returncode != 0:
        raise RuntimeError(
            "Standalone reward plugin import failed: "
            + (result.stderr.strip() or result.stdout.strip())
        )
    return {
        "status": "PASS",
        "plugin": str(plugin),
        "import_mode": "standalone_plugin_directory_only",
        "repo_root_required": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plugin",
        type=Path,
        default=Path("rl/reward_builder_v2.py"),
    )
    args = parser.parse_args()
    try:
        result = validate_standalone_import(args.plugin)
        print(
            json.dumps(
                {"event": "reward_plugin_boundary_check", **result},
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "event": "reward_plugin_boundary_check",
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
