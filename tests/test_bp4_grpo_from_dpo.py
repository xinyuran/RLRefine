import json
import tempfile
import unittest
from pathlib import Path

from evaluation.keyword_evaluator import file_sha256
from scripts.bp2_pipeline import (
    build_stage_command,
    load_config,
    read_authorization,
    validate_config,
)


CONFIG_PATH = Path("configs/experiments/bp2_e4_grpo_from_dpo_v1.yaml")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class Bp4GrpoFromDpoTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG_PATH)

    def test_e4_config_preserves_global_batch_and_split_policy(self):
        contract = validate_config(
            self.config, Path("."), require_bp1_assets=False
        )
        self.assertEqual(contract["pipeline_id"], "bp2_e4_grpo_from_dpo_v1")
        grpo = self.config["stages"]["grpo"]
        self.assertEqual(
            grpo["args"]["per_device_train_batch_size"]
            * grpo["args"]["gradient_accumulation_steps"],
            32,
        )
        self.assertEqual(grpo["args"]["num_generations"], 4)
        self.assertEqual(grpo["args"]["per_device_eval_batch_size"], 4)
        self.assertTrue(grpo["enforce_generation_batch_divisibility"])
        self.assertEqual(
            self.config["evaluation"]["forbidden_selection_split"],
            "frozen_human_gold",
        )

    def test_e4_config_rejects_incompatible_eval_batch(self):
        self.config["stages"]["grpo"]["args"]["per_device_eval_batch_size"] = 1
        with self.assertRaisesRegex(
            ValueError, "global eval batch size must be divisible"
        ):
            validate_config(self.config, Path("."), require_bp1_assets=False)

    def test_e4_config_rejects_incompatible_train_batch(self):
        grpo_args = self.config["stages"]["grpo"]["args"]
        grpo_args["per_device_train_batch_size"] = 2
        grpo_args["gradient_accumulation_steps"] = 16
        with self.assertRaisesRegex(
            ValueError, "global train batch size must be divisible"
        ):
            validate_config(self.config, Path("."), require_bp1_assets=False)

    def test_e4_command_uses_base_with_dpo_policy_and_reference_adapters(self):
        checkpoint = "/checkpoints/dpo-74"
        command, env = build_stage_command(
            self.config,
            "grpo",
            authorization={"model_path": checkpoint},
        )
        self.assertEqual(command[:2], ["swift", "rlhf"])
        self.assertEqual(command[command.index("--rlhf_type") + 1], "grpo")
        self.assertEqual(
            command[command.index("--model") + 1],
            self.config["model"]["base_path"],
        )
        self.assertEqual(command[command.index("--adapters") + 1], checkpoint)
        self.assertEqual(
            command[command.index("--ref_adapters") + 1], checkpoint
        )
        self.assertEqual(command[command.index("--lora_rank") + 1], "16")
        self.assertEqual(command[command.index("--lora_alpha") + 1], "64")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "7")

    def _authorized_fixture(self, root: Path) -> tuple[Path, Path]:
        grpo = self.config["stages"]["grpo"]
        train_path = root / grpo["dataset"]
        dev_path = root / grpo["val_dataset"]
        write_jsonl(train_path, [{"sample_id": "train", "solution": "{}"}])
        write_jsonl(dev_path, [{"sample_id": "dev", "solution": "{}"}])

        reward_gate = root / grpo["dependencies"][0]
        write_json(reward_gate, {"status": "PASS", "checks": {"reward": True}})
        preference_gate = root / grpo["dependencies"][1]
        write_json(
            preference_gate,
            {
                "status": "PASS",
                "checks": {"lineage": True},
                "counts": {"grpo_train": 2360, "grpo_dev": 331},
                "files": {
                    "grpo_train": {
                        "path": grpo["dataset"],
                        "sha256": file_sha256(train_path),
                    },
                    "grpo_dev": {
                        "path": grpo["val_dataset"],
                        "sha256": file_sha256(dev_path),
                    },
                },
            },
        )

        checkpoint = root / "checkpoint-74"
        checkpoint.mkdir()
        trainer_state = checkpoint / "trainer_state.json"
        trainer_state.write_text(
            json.dumps({"global_step": 74}), encoding="utf-8"
        )
        (checkpoint / "adapter_config.json").write_text(
            "{}", encoding="utf-8"
        )
        (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
        amendment_path = root / grpo["dependencies"][2]
        amendment = {
            "report_version":
                "bp4-manual-research-progression-amendment-v1",
            "status": "PASS",
            "decision": "ACCEPT_DPO_AS_GRPO_RESEARCH_UPSTREAM",
            "historical_research_decision":
                "REJECT_DPO_RESEARCH_PROGRESSION",
            "historical_deployment_decision": "REJECT_DPO_DEPLOYMENT",
            "candidate_checkpoint": str(checkpoint),
            "paired_packet_sha256": "p" * 64,
            "historical_gate_sha256": "g" * 64,
            "checks": {"history_preserved": True},
        }
        write_json(amendment_path, amendment)

        plugin = root / grpo["args"]["external_plugins"]
        plugin.parent.mkdir(parents=True, exist_ok=True)
        plugin.write_text("# reward plugin\n", encoding="utf-8")

        auth_path = root / grpo["authorization_file"]
        write_json(
            auth_path,
            {
                "authorization_version":
                    "bp4-e4-grpo-from-dpo-active-v1",
                "stage": "grpo",
                "decision": "AUTHORIZE_GRPO_TRAINING",
                "activation_status": "ACTIVE_APPROVED",
                "config_sha256": "expected",
                "model_path": str(checkpoint),
                "dataset_hash_source_path": str(preference_gate),
                "reward_plugin_sha256": file_sha256(plugin),
                "upstream_acceptance_path": str(amendment_path),
                "upstream_acceptance_sha256": file_sha256(amendment_path),
                "upstream_paired_packet_sha256": "p" * 64,
                "upstream_historical_gate_sha256": "g" * 64,
                "upstream_trainer_state_sha256": file_sha256(trainer_state),
            },
        )
        return train_path, amendment_path

    def test_e4_authorization_binds_server_bp3_hashes_and_manual_amendment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._authorized_fixture(root)
            _, blockers = read_authorization(
                self.config,
                "grpo",
                root,
                config_sha256="expected",
            )
            self.assertEqual(blockers, [])

    def test_e4_authorization_rejects_data_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path, _ = self._authorized_fixture(root)
            train_path.write_text('{"sample_id":"changed"}\n', encoding="utf-8")
            _, blockers = read_authorization(
                self.config,
                "grpo",
                root,
                config_sha256="expected",
            )
            self.assertIn("dataset_authorization_hash_mismatch", blockers)

    def test_e4_authorization_rejects_reward_plugin_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._authorized_fixture(root)
            plugin = (
                root
                / self.config["stages"]["grpo"]["args"]["external_plugins"]
            )
            plugin.write_text("# changed reward plugin\n", encoding="utf-8")
            _, blockers = read_authorization(
                self.config,
                "grpo",
                root,
                config_sha256="expected",
            )
            self.assertIn(
                "reward_plugin_authorization_hash_mismatch", blockers
            )

    def test_e4_authorization_rejects_amendment_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, amendment_path = self._authorized_fixture(root)
            amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
            amendment["decision"] = "REJECT_DPO_AS_GRPO_RESEARCH_UPSTREAM"
            write_json(amendment_path, amendment)
            _, blockers = read_authorization(
                self.config,
                "grpo",
                root,
                config_sha256="expected",
            )
            self.assertIn(
                "authorization_evidence_hash_mismatch:upstream_acceptance",
                blockers,
            )

    def test_runner_launches_only_e4_grpo(self):
        script = Path(
            "scripts/run_bp4_e4_grpo_from_dpo_training.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--stage grpo --dry-run --require-authorized", script)
        self.assertIn("--stage grpo", script)
        self.assertNotIn("--stage dpo", script)
        self.assertIn("REWARD_V2_METRICS", script)
        self.assertIn('"human_gold_used":false', script)


if __name__ == "__main__":
    unittest.main()
