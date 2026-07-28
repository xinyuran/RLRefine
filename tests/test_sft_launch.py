import json
import unittest
from pathlib import Path

import yaml

from evaluation.keyword_evaluator import file_sha256
from scripts.prepare_experiment_manifest import _git_lineage, _source_lineage
from scripts.validate_sft_launch import source_diff, validate_launch
from tests import test_experiment_manifest as fixture_module


class SftLaunchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.ExperimentManifestTests(
            methodName="test_valid_config_locks_disjoint_train_and_dev"
        )
        self.fixture.setUp()
        self.root = self.fixture.root
        self.config = self.fixture.config
        config_dir = self.root / "configs"
        config_dir.mkdir()
        reports_dir = self.root / "reports" / "lineage" / "p1_sft_lora_v1"
        reports_dir.mkdir(parents=True)
        scripts_dir = self.root / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "training_entry.py").write_text("VERSION = 1\n", encoding="utf-8")
        self.config_file = config_dir / "experiment.yaml"
        self.swift_file = reports_dir / "swift_sft_config.yaml"
        self.manifest_file = reports_dir / "run_manifest.json"
        self.config_file.write_text(yaml.safe_dump(self.config, sort_keys=True), encoding="utf-8")
        self.swift_file.write_text(yaml.safe_dump(self.config["swift"], sort_keys=True), encoding="utf-8")
        self.manifest = {
            "manifest_version": "structalign-run-manifest-v3",
            "status": "PASS",
            "experiment": self.config["experiment"],
            "config": {
                "path": self.config_file.relative_to(self.root).as_posix(),
                "sha256": file_sha256(self.config_file),
            },
            "swift_config": {
                "path": self.swift_file.relative_to(self.root).as_posix(),
                "sha256": file_sha256(self.swift_file),
                "command": [
                    "python", "-m", "scripts.swift_yaml_launcher", "--config",
                    self.swift_file.relative_to(self.root).as_posix(),
                ],
            },
            "source": _source_lineage(self.root),
            "git": _git_lineage(self.root),
        }
        self.manifest_file.write_text(json.dumps(self.manifest), encoding="utf-8")

    def tearDown(self):
        self.fixture.tearDown()

    def test_matching_manifest_allows_launch(self):
        result = validate_launch(self.manifest_file, self.root)
        self.assertEqual(result["experiment_id"], "p1_sft_lora_v1")
        self.assertEqual(result["source_sha256"], self.manifest["source"]["sha256"])

    def test_source_drift_blocks_launch(self):
        (self.root / "scripts" / "training_entry.py").write_text("VERSION = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, r'"modified": \["scripts/training_entry.py"\]'):
            validate_launch(self.manifest_file, self.root)

    def test_notebook_checkpoint_created_after_gate_does_not_block_launch(self):
        checkpoint_dir = self.root / "scripts" / ".ipynb_checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "training_entry-checkpoint.py").write_text(
            "VERSION = 'backup'\n", encoding="utf-8"
        )
        result = validate_launch(self.manifest_file, self.root)
        self.assertEqual(result["source_sha256"], self.manifest["source"]["sha256"])
        self.assertEqual(result["ignored_notebook_checkpoints"]["count"], 1)

    def test_source_diff_reports_added_removed_and_modified_files(self):
        frozen = {
            "file_count": 2,
            "sha256": "frozen",
            "files": [
                {"path": "scripts/removed.py", "sha256": "a"},
                {"path": "scripts/modified.py", "sha256": "b"},
            ],
        }
        current = {
            "file_count": 2,
            "sha256": "current",
            "files": [
                {"path": "scripts/modified.py", "sha256": "c"},
                {"path": "scripts/added.py", "sha256": "d"},
            ],
        }
        self.assertEqual(source_diff(frozen, current), {
            "added": ["scripts/added.py"],
            "removed": ["scripts/removed.py"],
            "modified": ["scripts/modified.py"],
            "frozen_file_count": 2,
            "current_file_count": 2,
            "frozen_sha256": "frozen",
            "current_sha256": "current",
        })


if __name__ == "__main__":
    unittest.main()
