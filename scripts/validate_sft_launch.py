"""Verify that an existing P0-4 manifest still matches the SFT launch state."""
import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from evaluation.keyword_evaluator import file_sha256
from scripts.prepare_experiment_manifest import _git_lineage, _source_lineage, validate_config


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True), flush=True)


def source_diff(frozen: dict, current: dict) -> dict:
    frozen_files = {item["path"]: item["sha256"] for item in frozen["files"]}
    current_files = {item["path"]: item["sha256"] for item in current["files"]}
    return {
        "added": sorted(current_files.keys() - frozen_files.keys()),
        "removed": sorted(frozen_files.keys() - current_files.keys()),
        "modified": sorted(
            path for path in current_files.keys() & frozen_files.keys()
            if current_files[path] != frozen_files[path]
        ),
        "frozen_file_count": frozen["file_count"],
        "current_file_count": current["file_count"],
        "frozen_sha256": frozen["sha256"],
        "current_sha256": current["sha256"],
    }


def validate_launch(manifest_file: Path, repo_root: Path) -> dict:
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != "structalign-run-manifest-v3" or manifest.get("status") != "PASS":
        raise ValueError("A passing v3 reproducibility manifest is required")

    config_path = repo_root / manifest["config"]["path"]
    swift_path = repo_root / manifest["swift_config"]["path"]
    if file_sha256(config_path) != manifest["config"]["sha256"]:
        raise ValueError("Experiment config changed after the reproducibility gate")
    if file_sha256(swift_path) != manifest["swift_config"]["sha256"]:
        raise ValueError("Generated Swift config changed after the reproducibility gate")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config, repo_root)

    current_source = _source_lineage(repo_root)
    if current_source["sha256"] != manifest["source"]["sha256"]:
        diff = source_diff(manifest["source"], current_source)
        raise ValueError(
            "Source tree changed after the reproducibility gate: "
            + json.dumps(diff, ensure_ascii=False, sort_keys=True)
        )
    current_git = _git_lineage(repo_root)
    frozen_git = manifest["git"]
    if current_git["available"] != frozen_git["available"]:
        raise ValueError("Git availability changed after the reproducibility gate")
    if current_git["available"] and current_git["commit"] != frozen_git["commit"]:
        raise ValueError("Git commit changed after the reproducibility gate")

    expected_command = [
        "python", "-m", "scripts.swift_yaml_launcher",
        "--config", manifest["swift_config"]["path"],
    ]
    if manifest["swift_config"]["command"] != expected_command:
        raise ValueError("Manifest Swift command is not the locked compatibility command")
    return {
        "experiment_id": manifest["experiment"]["id"],
        "source_sha256": current_source["sha256"],
        "ignored_notebook_checkpoints": current_source["ignored_notebook_checkpoints"],
        "git_commit": current_git["commit"],
        "swift_config": manifest["swift_config"]["path"],
        "output_dir": config["swift"]["output_dir"],
        "command": expected_command,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        result = validate_launch(args.manifest, args.repo_root.resolve())
        emit("p1_sft_launch_validation", status="PASS", **result)
        return 0
    except Exception as exc:
        emit("p1_sft_launch_validation", status="FAIL", error_type=type(exc).__name__, error=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
