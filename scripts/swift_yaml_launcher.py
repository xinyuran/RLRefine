"""Expand a frozen Swift YAML into explicit CLI arguments for ms-swift 3.11.x."""
import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True), flush=True)


def _cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    raise ValueError(f"Unsupported Swift YAML value type: {type(value).__name__}")


def build_launch(config: dict) -> tuple[list[str], dict[str, str]]:
    if not isinstance(config, dict):
        raise ValueError("Swift YAML must contain a mapping")
    raw_env = config.get("ENV")
    if not isinstance(raw_env, dict):
        raise ValueError("Swift YAML ENV must contain a mapping")
    if not config.get("model") or not config.get("dataset") or not config.get("val_dataset"):
        raise ValueError("Swift YAML must explicitly define model, dataset, and val_dataset")

    command = ["swift", "sft"]
    for key in sorted(config):
        if key == "ENV":
            continue
        command.extend([f"--{key}", _cli_value(config[key])])
    launch_env = {str(key): _cli_value(value) for key, value in sorted(raw_env.items())}
    return command, launch_env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        command, launch_env = build_launch(config)
        event = {
            "status": "PASS",
            "config": str(args.config),
            "command": command,
            "launch_env": launch_env,
            "dry_run": args.dry_run,
        }
        emit("swift_yaml_launcher", **event)
        if args.dry_run:
            return 0
        process_env = os.environ.copy()
        process_env.update(launch_env)
        return subprocess.run(command, env=process_env, check=False).returncode
    except Exception as exc:
        emit("swift_yaml_launcher", status="FAIL", error_type=type(exc).__name__, error=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
