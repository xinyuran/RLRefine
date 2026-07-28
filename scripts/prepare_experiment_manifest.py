"""Validate a controlled experiment config and freeze its execution lineage."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from evaluation.keyword_evaluator import file_sha256, write_json_atomic, write_text_atomic
from evaluation.baseline_inference import stable_hash


SCHEMA_VERSION = "structalign-experiment-v1"
CRITICAL_PACKAGES = ("ms-swift", "torch", "transformers", "pyyaml")
SOURCE_ROOTS = ("configs", "core", "evaluation", "examples", "prompts", "rl", "scripts", "tests")
SOURCE_SUFFIXES = {".json", ".py", ".sh", ".toml", ".yaml", ".yml"}
SOURCE_ROOT_FILES = ("requirements.txt", "pyproject.toml", "setup.cfg", "setup.py")


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True), flush=True)


def read_jsonl_contract(path: Path) -> Tuple[int, List[str], List[str], set]:
    sample_ids: List[str] = []
    seen_sample_ids = set()
    group_ids: List[str] = []
    splits = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            sample_id = row.get("sample_id")
            group_id = row.get("group_id")
            split = row.get("split")
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen_sample_ids:
                raise ValueError(f"Invalid or duplicate sample_id at {path}:{line_number}")
            seen_sample_ids.add(sample_id)
            if not isinstance(group_id, str) or not group_id:
                raise ValueError(f"Missing group_id at {path}:{line_number}")
            sample_ids.append(sample_id)
            group_ids.append(group_id)
            splits.add(split)
    return len(sample_ids), sample_ids, group_ids, splits


def read_unique_sample_ids(path: Path) -> set:
    sample_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
                raise ValueError(f"Invalid or duplicate sample_id at {path}:{line_number}")
            sample_ids.add(sample_id)
    return sample_ids


def _require_mapping(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config field {key!r} must be a mapping")
    return value


def validate_config(config: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    experiment = _require_mapping(config, "experiment")
    model = _require_mapping(config, "model")
    data = _require_mapping(config, "data")
    acceptance = _require_mapping(config, "acceptance")
    swift = _require_mapping(config, "swift")
    if experiment.get("id") != "p1_sft_lora_v1" or experiment.get("stage") != "sft":
        raise ValueError("Unexpected experiment identity")
    if experiment.get("selection_split") != "teacher_dev":
        raise ValueError("SFT selection must use teacher_dev")
    if experiment.get("final_test_policy") != "frozen_human_gold_only_after_dev_selection":
        raise ValueError("Human gold isolation policy is missing")
    if acceptance.get("policy") != "preregistered_teacher_dev_no_human_gold_tuning":
        raise ValueError("SFT acceptance policy is missing")
    if acceptance.get("baseline_variant") != "b1_structured_v3":
        raise ValueError("SFT must be compared with the frozen B1 dev baseline")
    baseline_metrics = _require_mapping(acceptance, "baseline_metrics")
    gates = _require_mapping(acceptance, "gates")
    expected_baseline = {
        "micro_f1": 0.5504899872177247,
        "macro_f1": 0.5410905455690168,
        "schema_valid_rate": 0.8761329305135952,
        "hallucination_keyword_rate": 0.027972027972027972,
        "mean_output_tokens": 259.8368580060423,
        "max_token_output_count": 5,
    }
    if baseline_metrics != expected_baseline:
        raise ValueError("Frozen B1 baseline metrics have drifted")
    if acceptance.get("baseline_prediction_sha256") != "deec1c9138efa09910c6cd202f92d2c1bf3979b0a9f09db197bef7a5f4d59abc":
        raise ValueError("Frozen B1 prediction hash has drifted")
    required_gates = {
        "paired_macro_f1_delta_min": 0.02,
        "paired_macro_f1_bootstrap_95_ci_low_gt": 0.0,
        "micro_f1_min": expected_baseline["micro_f1"],
        "schema_valid_rate_min": 0.90,
        "hallucination_keyword_rate_max": 0.03,
        "mean_output_tokens_max": expected_baseline["mean_output_tokens"],
        "max_token_output_count_max": 5,
    }
    if gates != required_gates:
        raise ValueError("Preregistered SFT acceptance gates have drifted")

    model_path = Path(str(model.get("path", "")))
    if not model_path.is_dir():
        raise ValueError(f"Model directory not found: {model_path}")
    if model_path.name != model.get("snapshot_revision") or swift.get("model") != str(model_path):
        raise ValueError("Model path and snapshot revision do not match")

    checked = {}
    indexed = {}
    for name in ("train", "dev"):
        contract = _require_mapping(data, name)
        relative_path = Path(str(contract.get("path", "")))
        if relative_path.is_absolute():
            raise ValueError(f"{name} dataset path must be repository-relative")
        path = repo_root / relative_path
        if not path.is_file():
            raise ValueError(f"Dataset file not found: {path}")
        actual_hash = file_sha256(path)
        if actual_hash != contract.get("sha256"):
            raise ValueError(f"{name} dataset SHA256 mismatch")
        rows, sample_ids, group_ids, splits = read_jsonl_contract(path)
        if rows != contract.get("rows") or splits != {contract.get("split")}:
            raise ValueError(f"{name} dataset row count or split contract mismatch")
        sample_hash = stable_hash("\n".join(sorted(sample_ids)))
        if sample_hash != contract.get("sample_ids_sha256"):
            raise ValueError(f"{name} sample IDs SHA256 mismatch")
        checked[name] = {
            "path": str(relative_path), "rows": rows, "sha256": actual_hash,
            "sample_ids_sha256": sample_hash,
        }
        indexed[name] = {"samples": set(sample_ids), "groups": set(group_ids)}
    if indexed["train"]["samples"] & indexed["dev"]["samples"]:
        raise ValueError("Train/dev sample overlap detected")
    if indexed["train"]["groups"] & indexed["dev"]["groups"]:
        raise ValueError("Train/dev group overlap detected")

    forbidden = _require_mapping(data, "forbidden_gold")
    forbidden_path = Path(str(forbidden.get("path", "")))
    if forbidden_path.is_absolute() or not (repo_root / forbidden_path).is_file():
        raise ValueError("Frozen human gold path must exist and be repository-relative")
    absolute_gold_path = repo_root / forbidden_path
    if file_sha256(absolute_gold_path) != forbidden.get("sha256"):
        raise ValueError("Frozen human gold SHA256 mismatch")
    if str(forbidden_path) in {str(swift.get("dataset")), str(swift.get("val_dataset"))}:
        raise ValueError("Frozen human gold must not be used by SFT")
    gold_sample_ids = read_unique_sample_ids(absolute_gold_path)
    if indexed["train"]["samples"] & gold_sample_ids:
        raise ValueError("Train/frozen-gold sample overlap detected")
    if indexed["dev"]["samples"] & gold_sample_ids:
        raise ValueError("Dev/frozen-gold sample overlap detected")

    expected_swift = {
        "model": str(model_path),
        "dataset": checked["train"]["path"],
        "val_dataset": checked["dev"]["path"],
        "split_dataset_ratio": 0.0,
        "data_seed": 42,
        "seed": 42,
        "train_type": "lora",
        "load_best_model_at_end": True,
    }
    for key, expected in expected_swift.items():
        if swift.get(key) != expected:
            raise ValueError(f"Swift field {key!r} must be {expected!r}")
    env = _require_mapping(swift, "ENV")
    visible_devices = [item for item in str(env.get("CUDA_VISIBLE_DEVICES", "")).split(",") if item]
    if not visible_devices or int(env.get("NPROC_PER_NODE", 0)) != len(visible_devices):
        raise ValueError("CUDA_VISIBLE_DEVICES and NPROC_PER_NODE do not match")
    effective_batch_size = (
        len(visible_devices)
        * int(swift.get("per_device_train_batch_size", 0))
        * int(swift.get("gradient_accumulation_steps", 0))
    )
    if effective_batch_size != 32:
        raise ValueError("Effective global training batch size must remain 32")
    if Path(str(swift.get("output_dir", ""))).is_absolute():
        raise ValueError("Training output_dir must be repository-relative")
    return {
        "datasets": checked,
        "overlaps": {
            "sample_train_dev": 0, "group_train_dev": 0,
            "sample_train_gold": 0, "sample_dev_gold": 0,
        },
        "forbidden_gold": {"path": str(forbidden_path), "sha256": forbidden["sha256"]},
        "effective_global_batch_size": effective_batch_size,
    }


def _git_lineage(repo_root: Path) -> Dict[str, Any]:
    git_marker = repo_root / ".git"
    if not git_marker.exists():
        return {
            "available": False,
            "reason": "repository_copy_without_git_metadata",
            "commit": None,
            "branch": None,
            "dirty": None,
            "status_porcelain": [],
        }

    def run(*args: str) -> str:
        command = ["git", *args]
        try:
            result = subprocess.run(
                command, cwd=repo_root, check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Git metadata exists but the git executable is unavailable") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip() or "<empty stderr>"
            raise RuntimeError(
                f"Git lineage command failed (exit {exc.returncode}): "
                f"{' '.join(command)}; stderr: {stderr}"
            ) from exc
        return result.stdout.strip()
    status = run("status", "--porcelain").splitlines()
    return {
        "available": True,
        "reason": None,
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status_porcelain": status,
    }


def _source_lineage(repo_root: Path) -> Dict[str, Any]:
    candidate_files = []
    ignored_notebook_checkpoint_files = []
    for root_name in SOURCE_ROOTS:
        root = repo_root / root_name
        if root.is_dir():
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                if ".ipynb_checkpoints" in path.parts:
                    ignored_notebook_checkpoint_files.append(path.relative_to(repo_root).as_posix())
                elif path.suffix.lower() in SOURCE_SUFFIXES and "__pycache__" not in path.parts:
                    candidate_files.append(path)
    for name in SOURCE_ROOT_FILES:
        path = repo_root / name
        if path.is_file():
            candidate_files.append(path)

    files = []
    for path in sorted(set(candidate_files), key=lambda item: item.relative_to(repo_root).as_posix()):
        relative_path = path.relative_to(repo_root).as_posix()
        files.append({"path": relative_path, "sha256": file_sha256(path)})
    if not files:
        raise ValueError("No source files found for source lineage snapshot")
    canonical = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "algorithm": "sha256-of-canonical-file-manifest-v1",
        "roots": list(SOURCE_ROOTS),
        "file_count": len(files),
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "files": files,
        "ignored_notebook_checkpoints": {
            "count": len(ignored_notebook_checkpoint_files),
            "paths": sorted(ignored_notebook_checkpoint_files),
            "policy": "excluded_from_source_lineage",
        },
    }


def _environment_lineage() -> Dict[str, Any]:
    packages = {}
    for name in CRITICAL_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    hardware: Dict[str, Any] = {"cuda_available": False, "cuda_devices": []}
    try:
        import torch
        hardware = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
            "torch_cuda": torch.version.cuda,
        }
    except ImportError:
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "hardware": hardware,
    }


def prepare_manifest(config_file: Path, output_dir: Path, repo_root: Path) -> Dict[str, Any]:
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a YAML mapping")
    contract = validate_config(config, repo_root)
    git_lineage = _git_lineage(repo_root)
    source_lineage = _source_lineage(repo_root)
    environment_lineage = _environment_lineage()
    swift_config = config["swift"]
    output_dir.mkdir(parents=True, exist_ok=True)
    swift_config_file = output_dir / "swift_sft_config.yaml"
    write_text_atomic(
        swift_config_file,
        yaml.safe_dump(swift_config, allow_unicode=True, sort_keys=True),
    )
    requirements = repo_root / "requirements.txt"
    manifest = {
        "manifest_version": "structalign-run-manifest-v3",
        "status": "PASS",
        "experiment": config["experiment"],
        "acceptance": config["acceptance"],
        "model": config["model"],
        "config": {"path": str(config_file), "sha256": file_sha256(config_file)},
        "swift_config": {
            "path": str(swift_config_file),
            "sha256": file_sha256(swift_config_file),
            "command": [
                "python", "-m", "scripts.swift_yaml_launcher",
                "--config", str(swift_config_file),
            ],
        },
        "data_contract": contract,
        "git": git_lineage,
        "source": source_lineage,
        "environment": environment_lineage,
        "requirements": {
            "path": str(requirements),
            "sha256": file_sha256(requirements) if requirements.is_file() else None,
        },
    }
    manifest_file = output_dir / "run_manifest.json"
    write_json_atomic(manifest_file, manifest)
    emit(
        "p0_reproducibility_manifest_complete",
        status="PASS",
        experiment_id=config["experiment"]["id"],
        manifest_file=str(manifest_file),
        manifest_sha256=file_sha256(manifest_file),
        swift_config_file=str(swift_config_file),
        swift_config_sha256=file_sha256(swift_config_file),
        train=contract["datasets"]["train"],
        dev=contract["datasets"]["dev"],
        overlaps=contract["overlaps"],
        forbidden_gold=contract["forbidden_gold"],
        git_commit=manifest["git"]["commit"],
        git_available=manifest["git"]["available"],
        git_dirty=manifest["git"]["dirty"],
        source_file_count=manifest["source"]["file_count"],
        source_sha256=manifest["source"]["sha256"],
        ignored_notebook_checkpoints=manifest["source"]["ignored_notebook_checkpoints"],
        environment=manifest["environment"],
        command=manifest["swift_config"]["command"],
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        prepare_manifest(args.config, args.output_dir, args.repo_root.resolve())
        return 0
    except Exception as exc:
        emit(
            "p0_reproducibility_manifest_complete", status="FAIL",
            error_type=type(exc).__name__, error=str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
