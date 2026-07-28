import json
import tempfile
import unittest
from pathlib import Path

import yaml
import tests.test_bp4_base_v2 as base_fixture

from evaluation.bp4_base_v2 import package_evaluation as package_base
from evaluation.bp4_sft_v2 import (
    CHALLENGE_PREDICTIONS,
    DEV_PREDICTIONS,
    GATE,
    package_evaluation,
    paired_bootstrap,
    select_best_checkpoint,
)
from evaluation.keyword_evaluator import file_sha256, read_jsonl, write_jsonl_atomic


class Bp4SftV2Tests(unittest.TestCase):
    def test_bootstrap_is_paired_deterministic_and_reports_wins(self):
        first = paired_bootstrap(
            [0.0, 0.5, 1.0],
            [0.5, 0.5, 1.0],
            iterations=10000,
            seed=42,
        )
        second = paired_bootstrap(
            [0.0, 0.5, 1.0],
            [0.5, 0.5, 1.0],
            iterations=10000,
            seed=42,
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(
            first["macro_f1_delta_candidate_minus_baseline"], 1 / 6
        )
        self.assertEqual(first["candidate_wins"], 1)
        self.assertEqual(first["baseline_wins"], 0)
        self.assertEqual(first["ties"], 2)

    def test_checkpoint_selection_uses_recorded_best_not_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            best = root / "training/checkpoint-100"
            latest = root / "training/checkpoint-200"
            for checkpoint in (best, latest):
                checkpoint.mkdir(parents=True)
                (checkpoint / "trainer_state.json").write_text(
                    '{"global_step": 100}', encoding="utf-8"
                )
                (checkpoint / "adapter_config.json").write_text(
                    "{}", encoding="utf-8"
                )
            report = root / "artifacts.json"
            report.write_text(
                json.dumps(
                    {
                        "latest_checkpoint": {"checkpoint": str(latest)},
                        "best_model_checkpoint": str(best),
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(select_best_checkpoint(root, report), best)

    def test_checkpoint_without_adapter_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint-100"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.json").write_text(
                '{"global_step": 100}', encoding="utf-8"
            )
            report = root / "artifacts.json"
            report.write_text(
                json.dumps({"best_model_checkpoint": str(checkpoint)}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "adapter_config"):
                select_best_checkpoint(root, report)

    def test_complete_package_records_valid_negative_result_without_downstream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = base_fixture.Bp4BaseV2Tests()
            config_path, task_path, _, task = helper.make_repo(root)
            package_base(root, config_path, task_path)
            bp2_path = (
                root / "configs/experiments/bp2_post_training_pipeline_v1.yaml"
            )
            bp2 = yaml.safe_load(bp2_path.read_text(encoding="utf-8"))
            train_path = root / bp2["stages"]["sft"]["dataset"]
            write_jsonl_atomic(train_path, [{"sample_id": "train"}])
            base_packet = root / task["required_outputs"]["base_packet"]
            auth_path = root / bp2["stages"]["sft"]["authorization_file"]
            auth_path.parent.mkdir(parents=True, exist_ok=True)
            auth_path.write_text(
                json.dumps(
                    {
                        "stage": "sft",
                        "decision": "AUTHORIZE_SFT_TRAINING",
                        "activation_status": "ACTIVE_APPROVED",
                        "approved_scope": "BP4_CONTROLLED_SFT_ONLY",
                        "downstream_training_authorized": False,
                        "config_sha256": file_sha256(bp2_path),
                        "dataset_sha256": file_sha256(train_path),
                        "val_dataset_sha256": file_sha256(
                            root / bp2["stages"]["sft"]["val_dataset"]
                        ),
                        "base_packet_path": str(base_packet),
                        "base_packet_sha256": file_sha256(base_packet),
                    }
                ),
                encoding="utf-8",
            )
            checkpoint = root / "reports/training/bp2_sft_v2/checkpoint-100"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer_state.json").write_text(
                '{"global_step":100}', encoding="utf-8"
            )
            (checkpoint / "adapter_config.json").write_text(
                "{}", encoding="utf-8"
            )
            artifacts = root / "reports/bp4_controlled_matrix/sft/artifacts.json"
            artifacts.parent.mkdir(parents=True, exist_ok=True)
            artifacts.write_text(
                json.dumps({"best_model_checkpoint": str(checkpoint)}),
                encoding="utf-8",
            )
            write_jsonl_atomic(
                root / DEV_PREDICTIONS,
                read_jsonl(root / task["required_outputs"]["dev_predictions"]),
            )
            write_jsonl_atomic(
                root / CHALLENGE_PREDICTIONS,
                read_jsonl(
                    root / task["required_outputs"]["challenge_predictions"]
                ),
            )
            gate = package_evaluation(root, config_path, artifacts)
            self.assertEqual(gate["status"], "PASS")
            self.assertEqual(gate["decision"], "REJECT_SFT_CANDIDATE")
            self.assertFalse(gate["downstream_training_authorized"])
            self.assertFalse(
                (root / "reports/model_gates/bp2_sft_acceptance.json").exists()
            )
            self.assertEqual(
                json.loads((root / GATE).read_text(encoding="utf-8"))[
                    "next_execution"
                ],
                "STOP_POST_TRAINING_AT_BASE_V2",
            )


if __name__ == "__main__":
    unittest.main()
