import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.baseline_inference import stable_hash
from evaluation.keyword_evaluator import file_sha256
from scripts.prepare_experiment_manifest import _git_lineage, _source_lineage, validate_config


class ExperimentManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / "snapshots" / "revision-1"
        self.model.mkdir(parents=True)
        (self.root / "data").mkdir()
        self.train = self.root / "data" / "train.jsonl"
        self.dev = self.root / "data" / "dev.jsonl"
        self.gold = self.root / "data" / "gold.jsonl"
        self._write_rows(self.train, [("train-1", "group-1", "train")])
        self._write_rows(self.dev, [("dev-1", "group-2", "dev")])
        self.gold.write_text(json.dumps({"sample_id": "gold-1"}) + "\n", encoding="utf-8")
        self.config = self._config()

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _write_rows(path, rows):
        path.write_text("".join(
            json.dumps({"sample_id": sample, "group_id": group, "split": split}) + "\n"
            for sample, group, split in rows
        ), encoding="utf-8")

    def _data_contract(self, path, sample_id, split):
        return {
            "path": str(path.relative_to(self.root)),
            "rows": 1,
            "sha256": file_sha256(path),
            "sample_ids_sha256": stable_hash(sample_id),
            "split": split,
        }

    def _config(self):
        train = self._data_contract(self.train, "train-1", "train")
        dev = self._data_contract(self.dev, "dev-1", "dev")
        return {
            "schema_version": "structalign-experiment-v1",
            "experiment": {
                "id": "p1_sft_lora_v1",
                "stage": "sft",
                "selection_split": "teacher_dev",
                "final_test_policy": "frozen_human_gold_only_after_dev_selection",
            },
            "acceptance": {
                "policy": "preregistered_teacher_dev_no_human_gold_tuning",
                "baseline_variant": "b1_structured_v3",
                "baseline_prediction_sha256": "deec1c9138efa09910c6cd202f92d2c1bf3979b0a9f09db197bef7a5f4d59abc",
                "baseline_metrics": {
                    "micro_f1": 0.5504899872177247,
                    "macro_f1": 0.5410905455690168,
                    "schema_valid_rate": 0.8761329305135952,
                    "hallucination_keyword_rate": 0.027972027972027972,
                    "mean_output_tokens": 259.8368580060423,
                    "max_token_output_count": 5,
                },
                "gates": {
                    "paired_macro_f1_delta_min": 0.02,
                    "paired_macro_f1_bootstrap_95_ci_low_gt": 0.0,
                    "micro_f1_min": 0.5504899872177247,
                    "schema_valid_rate_min": 0.90,
                    "hallucination_keyword_rate_max": 0.03,
                    "mean_output_tokens_max": 259.8368580060423,
                    "max_token_output_count_max": 5,
                },
            },
            "model": {
                "path": str(self.model),
                "snapshot_revision": "revision-1",
            },
            "data": {
                "train": train,
                "dev": dev,
                "forbidden_gold": {
                    "path": str(self.gold.relative_to(self.root)),
                    "sha256": file_sha256(self.gold),
                },
            },
            "swift": {
                "ENV": {"CUDA_VISIBLE_DEVICES": "0,1", "NPROC_PER_NODE": "2"},
                "model": str(self.model),
                "dataset": train["path"],
                "val_dataset": dev["path"],
                "split_dataset_ratio": 0.0,
                "data_seed": 42,
                "seed": 42,
                "train_type": "lora",
                "load_best_model_at_end": True,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "output_dir": "reports/training/test",
            },
        }

    def test_valid_config_locks_disjoint_train_and_dev(self):
        result = validate_config(self.config, self.root)
        self.assertEqual(result["datasets"]["train"]["rows"], 1)
        self.assertEqual(result["overlaps"], {
            "sample_train_dev": 0,
            "group_train_dev": 0,
            "sample_train_gold": 0,
            "sample_dev_gold": 0,
        })

    def test_dataset_hash_mismatch_is_rejected(self):
        self.config["data"]["train"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            validate_config(self.config, self.root)

    def test_group_overlap_is_rejected(self):
        self._write_rows(self.dev, [("dev-1", "group-1", "dev")])
        self.config["data"]["dev"] = self._data_contract(self.dev, "dev-1", "dev")
        with self.assertRaisesRegex(ValueError, "group overlap"):
            validate_config(self.config, self.root)

    def test_gold_cannot_be_used_as_training_data(self):
        self.config["swift"]["dataset"] = self.config["data"]["forbidden_gold"]["path"]
        with self.assertRaisesRegex(ValueError, "must not be used"):
            validate_config(self.config, self.root)

    def test_gold_sample_overlap_is_rejected(self):
        self.gold.write_text(json.dumps({"sample_id": "train-1"}) + "\n", encoding="utf-8")
        self.config["data"]["forbidden_gold"]["sha256"] = file_sha256(self.gold)
        with self.assertRaisesRegex(ValueError, "Train/frozen-gold sample overlap"):
            validate_config(self.config, self.root)

    def test_gpu_process_count_must_match(self):
        self.config["swift"]["ENV"]["NPROC_PER_NODE"] = "1"
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_config(self.config, self.root)

    def test_effective_global_batch_size_must_remain_locked(self):
        self.config["swift"]["gradient_accumulation_steps"] = 3
        with self.assertRaisesRegex(ValueError, "Effective global training batch size"):
            validate_config(self.config, self.root)

    def test_repository_copy_without_git_uses_explicit_unavailable_state(self):
        lineage = _git_lineage(self.root)
        self.assertFalse(lineage["available"])
        self.assertEqual(lineage["reason"], "repository_copy_without_git_metadata")
        self.assertIsNone(lineage["commit"])

    def test_source_lineage_changes_when_source_changes(self):
        source_dir = self.root / "scripts"
        source_dir.mkdir()
        source_file = source_dir / "runner.py"
        source_file.write_text("value = 1\n", encoding="utf-8")
        before = _source_lineage(self.root)
        source_file.write_text("value = 2\n", encoding="utf-8")
        after = _source_lineage(self.root)
        self.assertEqual(before["file_count"], after["file_count"])
        self.assertNotEqual(before["sha256"], after["sha256"])

    def test_source_lineage_ignores_notebook_checkpoint_files(self):
        source_dir = self.root / "scripts"
        source_dir.mkdir()
        (source_dir / "runner.py").write_text("VERSION = 1\n", encoding="utf-8")
        before = _source_lineage(self.root)
        checkpoint_dir = self.root / "scripts" / ".ipynb_checkpoints"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_file = checkpoint_dir / "runner-checkpoint.sh"
        checkpoint_file.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        after = _source_lineage(self.root)
        self.assertEqual(before["sha256"], after["sha256"])
        self.assertEqual(before["file_count"], after["file_count"])
        self.assertEqual(after["ignored_notebook_checkpoints"], {
            "count": 1,
            "paths": ["scripts/.ipynb_checkpoints/runner-checkpoint.sh"],
            "policy": "excluded_from_source_lineage",
        })

    def test_git_failure_reports_stderr_when_metadata_exists(self):
        (self.root / ".git").mkdir()
        failure = __import__("subprocess").CalledProcessError(
            128, ["git", "status", "--porcelain"], stderr="fatal: dubious ownership"
        )
        with patch("scripts.prepare_experiment_manifest.subprocess.run", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "dubious ownership"):
                _git_lineage(self.root)


if __name__ == "__main__":
    unittest.main()
