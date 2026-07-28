import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import yaml

from evaluation.keyword_evaluator import file_sha256
from scripts.bp2_pipeline import (
    build_stage_command,
    evaluation_task,
    inspect_artifacts,
    load_config,
    main,
    prepare_package,
    read_authorization,
    validate_config,
    validate_resume_checkpoint,
)


REAL_CONFIG = Path("configs/experiments/bp2_post_training_pipeline_v1.yaml")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class Bp2PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(REAL_CONFIG)

    def test_all_stage_commands_use_expected_ms_swift_entrypoints(self):
        sft, _ = build_stage_command(self.config, "sft")
        dpo, _ = build_stage_command(self.config, "dpo")
        grpo, _ = build_stage_command(self.config, "grpo")
        self.assertEqual(sft[:2], ["swift", "sft"])
        self.assertEqual(dpo[:2], ["swift", "rlhf"])
        self.assertEqual(dpo[dpo.index("--rlhf_type") + 1], "dpo")
        self.assertEqual(grpo[:2], ["swift", "rlhf"])
        self.assertIn("grpo", grpo)
        self.assertIn("<AUTHORIZED_SFT_CHECKPOINT>", dpo)
        self.assertIn("<AUTHORIZED_UPSTREAM_CHECKPOINT>", grpo)

    def test_resume_is_explicit_and_not_enabled_by_default(self):
        base, _ = build_stage_command(self.config, "sft")
        resumed, _ = build_stage_command(
            self.config, "sft", resume_from="checkpoint-100"
        )
        self.assertNotIn("--resume_from_checkpoint", base)
        index = resumed.index("--resume_from_checkpoint")
        self.assertEqual(resumed[index + 1], "checkpoint-100")

    def test_config_rejects_human_gold_as_stage_data(self):
        config = deepcopy(self.config)
        config["data"]["train"]["path"] = config["data"]["forbidden_gold"]["path"]
        config["stages"]["sft"]["dataset"] = config["data"]["forbidden_gold"]["path"]
        with self.assertRaisesRegex(ValueError, "human gold"):
            validate_config(config, Path("."), require_bp1_assets=False)

    def test_missing_authorization_and_dependencies_block_execution(self):
        authorization, blockers = read_authorization(
            self.config,
            "dpo",
            Path("."),
            config_sha256=file_sha256(REAL_CONFIG),
        )
        self.assertIsNone(authorization)
        self.assertTrue(any(item.startswith("missing_authorization:") for item in blockers))
        self.assertTrue(any(item.startswith("missing_dependency:") for item in blockers))

    def test_required_authorized_dry_run_fails_on_blockers(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "sys.argv",
                [
                    "bp2_pipeline",
                    "launch",
                    "--config",
                    str(REAL_CONFIG.resolve()),
                    "--stage",
                    "dpo",
                    "--repo-root",
                    temporary,
                    "--dry-run",
                    "--require-authorized",
                ],
            ):
                self.assertEqual(main(), 2)

    def test_authorization_is_bound_to_config_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            auth_path = root / self.config["stages"]["sft"]["authorization_file"]
            auth_path.parent.mkdir(parents=True)
            auth_path.write_text(
                json.dumps(
                    {
                        "stage": "sft",
                        "decision": "AUTHORIZE_SFT_TRAINING",
                        "config_sha256": "wrong",
                    }
                ),
                encoding="utf-8",
            )
            train_path = root / self.config["stages"]["sft"]["dataset"]
            dev_path = root / self.config["stages"]["sft"]["val_dataset"]
            write_jsonl(train_path, [{"sample_id": "a"}])
            write_jsonl(dev_path, [{"sample_id": "b"}])
            _, blockers = read_authorization(
                self.config, "sft", root, config_sha256="expected"
            )
            self.assertIn("authorization_config_hash_mismatch", blockers)
            self.assertIn("authorization_not_active", blockers)

    def test_active_sft_authorization_requires_bound_base_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path = root / self.config["stages"]["sft"]["dataset"]
            dev_path = root / self.config["stages"]["sft"]["val_dataset"]
            write_jsonl(train_path, [{"sample_id": "train"}])
            write_jsonl(dev_path, [{"sample_id": "dev"}])
            packet_path = root / "reports/bp4/base_packet.json"
            packet_path.parent.mkdir(parents=True, exist_ok=True)
            packet_path.write_text(
                json.dumps(
                    {
                        "packet_version": "bp4-base-evaluation-v1",
                        "status": "PASS",
                        "dev_rows": 331,
                        "challenge_rows": 140,
                        "human_gold_used": False,
                    }
                ),
                encoding="utf-8",
            )
            auth_path = (
                root / self.config["stages"]["sft"]["authorization_file"]
            )
            auth_path.parent.mkdir(parents=True, exist_ok=True)
            auth_path.write_text(
                json.dumps(
                    {
                        "stage": "sft",
                        "decision": "AUTHORIZE_SFT_TRAINING",
                        "activation_status": "ACTIVE_APPROVED",
                        "config_sha256": "expected",
                        "dataset_sha256": file_sha256(train_path),
                        "val_dataset_sha256": file_sha256(dev_path),
                        "base_packet_path": str(packet_path),
                        "base_packet_sha256": file_sha256(packet_path),
                    }
                ),
                encoding="utf-8",
            )
            _, blockers = read_authorization(
                self.config, "sft", root, config_sha256="expected"
            )
            self.assertEqual(blockers, [])

    def test_rejected_model_gate_cannot_satisfy_downstream_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path = root / self.config["stages"]["dpo"]["dataset"]
            dev_path = root / self.config["stages"]["dpo"]["val_dataset"]
            write_jsonl(train_path, [{"sample_id": "train"}])
            write_jsonl(dev_path, [{"sample_id": "dev"}])
            model_path = root / "checkpoint-100"
            model_path.mkdir()
            preference = root / self.config["stages"]["dpo"]["dependencies"][0]
            preference.parent.mkdir(parents=True, exist_ok=True)
            preference.write_text('{"status":"PASS"}', encoding="utf-8")
            model_gate = root / self.config["stages"]["dpo"]["dependencies"][1]
            model_gate.parent.mkdir(parents=True, exist_ok=True)
            model_gate.write_text(
                json.dumps(
                    {
                        "report_version": "bp4-stage-acceptance-v1",
                        "status": "PASS",
                        "decision": "REJECT_SFT_CANDIDATE",
                        "checks": {"macro_gain": False},
                    }
                ),
                encoding="utf-8",
            )
            auth_path = root / self.config["stages"]["dpo"]["authorization_file"]
            auth_path.parent.mkdir(parents=True, exist_ok=True)
            auth_path.write_text(
                json.dumps(
                    {
                        "stage": "dpo",
                        "decision": "AUTHORIZE_DPO_TRAINING",
                        "activation_status": "ACTIVE_APPROVED",
                        "config_sha256": "expected",
                        "dataset_sha256": file_sha256(train_path),
                        "val_dataset_sha256": file_sha256(dev_path),
                        "model_path": str(model_path),
                        "upstream_acceptance_path": str(model_gate),
                        "upstream_acceptance_sha256": file_sha256(model_gate),
                    }
                ),
                encoding="utf-8",
            )
            _, blockers = read_authorization(
                self.config, "dpo", root, config_sha256="expected"
            )
            self.assertIn(
                "dependency_not_accepted:"
                + self.config["stages"]["dpo"]["dependencies"][1],
                blockers,
            )

    def test_two_level_research_gate_can_authorize_repair_dpo(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = deepcopy(self.config)
            dependency = "reports/model_gates/bp2_sft_two_level_amendment.json"
            config["stages"]["dpo"]["dependencies"][1] = dependency
            train_path = root / config["stages"]["dpo"]["dataset"]
            dev_path = root / config["stages"]["dpo"]["val_dataset"]
            write_jsonl(train_path, [{"sample_id": "train"}])
            write_jsonl(dev_path, [{"sample_id": "dev"}])
            model_path = root / "checkpoint-370"
            model_path.mkdir()
            preference = root / config["stages"]["dpo"]["dependencies"][0]
            preference.parent.mkdir(parents=True, exist_ok=True)
            preference.write_text('{"status":"PASS"}', encoding="utf-8")
            amendment_path = root / dependency
            amendment_path.parent.mkdir(parents=True, exist_ok=True)
            amendment_path.write_text(
                json.dumps(
                    {
                        "report_version": "bp4-two-level-gate-amendment-v1",
                        "status": "PASS",
                        "research_progression_decision":
                            "ACCEPT_SFT_RESEARCH_PROGRESSION",
                        "research_progression_checks": {"credible_gain": True},
                        "deployment_decision": "REJECT_SFT_DEPLOYMENT",
                        "paired_packet_sha256": "packet",
                        "legacy_acceptance_sha256": "acceptance",
                        "candidate_checkpoint": str(model_path),
                    }
                ),
                encoding="utf-8",
            )
            auth_path = root / config["stages"]["dpo"]["authorization_file"]
            auth_path.parent.mkdir(parents=True, exist_ok=True)
            auth_path.write_text(
                json.dumps(
                    {
                        "stage": "dpo",
                        "decision": "AUTHORIZE_DPO_TRAINING",
                        "activation_status": "ACTIVE_APPROVED",
                        "config_sha256": "expected",
                        "dataset_sha256": file_sha256(train_path),
                        "val_dataset_sha256": file_sha256(dev_path),
                        "model_path": str(model_path),
                        "upstream_acceptance_path": str(amendment_path),
                        "authorization_version":
                            "bp4-reliability-repair-active-v2",
                        "upstream_paired_packet_sha256": "packet",
                        "upstream_legacy_acceptance_sha256": "acceptance",
                        "upstream_research_decision":
                            "ACCEPT_SFT_RESEARCH_PROGRESSION",
                    }
                ),
                encoding="utf-8",
            )
            _, blockers = read_authorization(
                config, "dpo", root, config_sha256="expected"
            )
            self.assertEqual(blockers, [])

            amendment_path.write_text(
                json.dumps(
                    json.loads(amendment_path.read_text(encoding="utf-8")),
                    indent=4,
                ),
                encoding="utf-8",
            )
            _, blockers = read_authorization(
                config, "dpo", root, config_sha256="expected"
            )
            self.assertEqual(blockers, [])

    def test_repair_dpo_rejects_wrong_frozen_source_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = deepcopy(self.config)
            dependency = "reports/model_gates/bp2_sft_two_level_amendment.json"
            config["stages"]["dpo"]["dependencies"][1] = dependency
            train_path = root / config["stages"]["dpo"]["dataset"]
            dev_path = root / config["stages"]["dpo"]["val_dataset"]
            write_jsonl(train_path, [{"sample_id": "train"}])
            write_jsonl(dev_path, [{"sample_id": "dev"}])
            model_path = root / "checkpoint-370"
            model_path.mkdir()
            preference = root / config["stages"]["dpo"]["dependencies"][0]
            preference.parent.mkdir(parents=True, exist_ok=True)
            preference.write_text('{"status":"PASS"}', encoding="utf-8")
            amendment_path = root / dependency
            amendment_path.parent.mkdir(parents=True, exist_ok=True)
            amendment_path.write_text(
                json.dumps({
                    "report_version": "bp4-two-level-gate-amendment-v1",
                    "status": "PASS",
                    "research_progression_decision":
                        "ACCEPT_SFT_RESEARCH_PROGRESSION",
                    "research_progression_checks": {"credible_gain": True},
                    "deployment_decision": "REJECT_SFT_DEPLOYMENT",
                    "paired_packet_sha256": "wrong",
                    "legacy_acceptance_sha256": "acceptance",
                    "candidate_checkpoint": str(model_path),
                }),
                encoding="utf-8",
            )
            auth_path = root / config["stages"]["dpo"]["authorization_file"]
            auth_path.parent.mkdir(parents=True, exist_ok=True)
            auth_path.write_text(
                json.dumps({
                    "authorization_version": "bp4-reliability-repair-active-v2",
                    "stage": "dpo",
                    "decision": "AUTHORIZE_DPO_TRAINING",
                    "activation_status": "ACTIVE_APPROVED",
                    "config_sha256": "expected",
                    "dataset_sha256": file_sha256(train_path),
                    "val_dataset_sha256": file_sha256(dev_path),
                    "model_path": str(model_path),
                    "upstream_acceptance_path": str(amendment_path),
                    "upstream_paired_packet_sha256": "packet",
                    "upstream_legacy_acceptance_sha256": "acceptance",
                    "upstream_research_decision":
                        "ACCEPT_SFT_RESEARCH_PROGRESSION",
                }),
                encoding="utf-8",
            )
            _, blockers = read_authorization(
                config, "dpo", root, config_sha256="expected"
            )
            self.assertIn(
                "authorization_evidence_not_accepted:upstream_acceptance",
                blockers,
            )

    def test_repair_dpo_uses_base_with_sft_and_reference_adapters(self):
        config = load_config(
            Path("configs/experiments/bp2_dpo_reliability_v1.yaml")
        )
        command, _ = build_stage_command(
            config,
            "dpo",
            authorization={"model_path": "/checkpoints/sft-370"},
        )
        self.assertEqual(command[:2], ["swift", "rlhf"])
        self.assertEqual(command[command.index("--rlhf_type") + 1], "dpo")
        self.assertEqual(
            command[command.index("--gradient_checkpointing") + 1],
            "false",
        )
        self.assertEqual(
            command[command.index("--use_logits_to_keep") + 1],
            "true",
        )
        self.assertEqual(
            command[command.index("--per_device_train_batch_size") + 1],
            "1",
        )
        self.assertEqual(
            command[command.index("--gradient_accumulation_steps") + 1],
            "32",
        )
        model_index = command.index("--model")
        adapters_index = command.index("--adapters")
        ref_adapters_index = command.index("--ref_adapters")
        self.assertEqual(
            command[model_index + 1], config["model"]["base_path"]
        )
        self.assertEqual(command[adapters_index + 1], "/checkpoints/sft-370")
        self.assertEqual(
            command[ref_adapters_index + 1], "/checkpoints/sft-370"
        )

    def test_checkpoint_and_artifact_inspection(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for step in (100, 200):
                checkpoint = output / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "trainer_state.json").write_text(
                    json.dumps(
                        {
                            "global_step": step,
                            "best_model_checkpoint": "checkpoint-200",
                            "best_metric": 0.5,
                            "log_history": [{"step": step}],
                        }
                    ),
                    encoding="utf-8",
                )
            resume = validate_resume_checkpoint(output / "checkpoint-100")
            report = inspect_artifacts(output)
            self.assertEqual(resume["global_step"], 100)
            self.assertTrue(report["latest_checkpoint"]["checkpoint"].endswith("checkpoint-200"))
            self.assertEqual(report["checkpoint_count"], 2)

    def test_artifact_inspection_supports_ms_swift_versioned_run_dir(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            run = output / "v0-20260724-141709"
            checkpoint = run / "checkpoint-370"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer_state.json").write_text(
                json.dumps(
                    {
                        "global_step": 370,
                        "best_model_checkpoint": str(checkpoint),
                        "best_metric": 0.25,
                        "log_history": [{"step": 370}],
                    }
                ),
                encoding="utf-8",
            )
            report = inspect_artifacts(output)
            self.assertEqual(Path(report["selected_run_dir"]), run)
            self.assertEqual(report["discovered_run_count"], 1)
            self.assertEqual(report["checkpoint_count"], 1)
            self.assertEqual(
                Path(report["best_model_checkpoint"]), checkpoint
            )

    def test_unified_evaluation_task_uses_dev_and_challenge_not_gold(self):
        task = evaluation_task(self.config, "sft", "checkpoint-100")
        self.assertEqual(task["selection"]["path"], self.config["data"]["dev"]["path"])
        self.assertEqual(
            task["diagnostic"]["path"], self.config["data"]["challenge"]["path"]
        )
        self.assertEqual(task["forbidden_for_selection"], "frozen_human_gold")

    def test_prepare_package_end_to_end_without_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / self.config["model"]["snapshot_revision"]
            model.mkdir()
            config = deepcopy(self.config)
            config["model"]["base_path"] = str(model)
            data_rows = {
                "train": [{"sample_id": "train"}],
                "dev": [{"sample_id": "dev"}],
                "challenge": [{"sample_id": "challenge"}],
                "forbidden_gold": [{"sample_id": "gold"}],
            }
            for name, rows in data_rows.items():
                path = root / config["data"][name]["path"]
                write_jsonl(path, rows)
                config["data"][name]["sha256"] = file_sha256(path)
                if name != "forbidden_gold":
                    config["data"][name]["rows"] = len(rows)
            bp1_path = root / config["data"]["bp1_manifest"]["path"]
            bp1_path.parent.mkdir(parents=True, exist_ok=True)
            bp1_path.write_text(
                json.dumps(
                    {
                        "dataset_version": "keyword-v2.0.0",
                        "target_contract_version": "keyword-json-target-v2",
                        "frozen_gold_overlap": 0,
                    }
                ),
                encoding="utf-8",
            )
            config["data"]["bp1_manifest"]["sha256"] = file_sha256(bp1_path)
            config_path = root / "configs" / "bp2.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            output = root / "reports" / "bp2"
            gate = prepare_package(config_path, output, root)
            self.assertEqual(gate["decision"], "ACCEPT_BP2_ENGINEERING")
            self.assertTrue((output / "bp2_manifest.json").is_file())
            self.assertTrue((output / "bp2_feedback.json").is_file())
            self.assertEqual(
                gate["stage_plan"]["sft"]["execution_status"], "BLOCKED"
            )


if __name__ == "__main__":
    unittest.main()
