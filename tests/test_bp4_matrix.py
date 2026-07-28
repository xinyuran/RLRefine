import json
import tempfile
import unittest
from pathlib import Path

import yaml

from evaluation.keyword_evaluator import file_sha256
from scripts.bp4_matrix import (
    build_sft_authorization_candidate,
    prepare_package,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def write_rows(path, count):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"sample_id": f"sample-{index}"}) + "\n"
            for index in range(count)
        ),
        encoding="utf-8",
    )


class Bp4MatrixTests(unittest.TestCase):
    def make_repo(self, root):
        root = Path(root)
        bp2_source = Path(
            "configs/experiments/bp2_post_training_pipeline_v1.yaml"
        )
        bp2_path = root / bp2_source
        bp2_path.parent.mkdir(parents=True, exist_ok=True)
        bp2_path.write_text(
            bp2_source.read_text(encoding="utf-8"), encoding="utf-8"
        )

        template = yaml.safe_load(
            Path(
                "configs/experiments/bp4_controlled_model_matrix_v1.yaml"
            ).read_text(encoding="utf-8")
        )
        for name in ("bp2_gate", "bp3_gate", "preference_gate", "reward_gate"):
            gate_path = root / template["pipeline"][name]["path"]
            write_json(
                gate_path,
                {
                    "status": "PASS",
                    "decision": f"ACCEPT_SYNTHETIC_{name.upper()}",
                },
            )
            template["pipeline"][name]["sha256"] = file_sha256(gate_path)

        for name, count in (("selection", 331), ("diagnostic", 140)):
            data_path = root / template["evaluation"][name]["path"]
            write_rows(data_path, count)
            template["evaluation"][name]["sha256"] = file_sha256(data_path)
        write_rows(root / "data/canonical/keyword_v2/train.jsonl", 10)

        config_path = root / "configs/experiments/bp4_test.yaml"
        config_path.write_text(
            yaml.safe_dump(template, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return config_path, template

    def test_prepare_makes_base_ready_and_keeps_training_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, config = self.make_repo(root)
            gate = prepare_package(root, config_path)
            self.assertEqual(
                gate["decision"], "ACCEPT_BP4_MATRIX_ENGINEERING"
            )
            self.assertFalse(gate["training_started"])

            plan = json.loads(
                (root / config["outputs"]["matrix_plan"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(plan["rows"][0]["status"], "READY")
            self.assertTrue(
                all(row["status"] == "BLOCKED" for row in plan["rows"][1:])
            )
            request = json.loads(
                (
                    root / config["outputs"]["sft_authorization_candidate"]
                ).read_text(encoding="utf-8")
            )
            self.assertNotIn("decision", request)
            self.assertEqual(
                request["activation_status"],
                "BLOCKED_MISSING_BASE_V2_EVALUATION",
            )

    def test_existing_active_authorization_fails_engineering_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, config = self.make_repo(root)
            write_json(
                root / "reports/authorizations/bp2_sft.json",
                {"decision": "AUTHORIZE_SFT_TRAINING"},
            )
            with self.assertRaisesRegex(ValueError, "BP4 gate failed"):
                prepare_package(root, config_path)

    def test_base_packet_only_creates_inactive_sft_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, config = self.make_repo(root)
            packet_path = root / "base_packet.json"
            write_json(
                packet_path,
                {
                    "packet_version": "bp4-base-evaluation-v1",
                    "status": "PASS",
                    "target_contract_version": config["evaluation"][
                        "target_contract_version"
                    ],
                    "dev_rows": 331,
                    "challenge_rows": 140,
                    "human_gold_used": False,
                    "challenge_usage": "diagnostic_only",
                    "selection_data_sha256": config["evaluation"]["selection"][
                        "sha256"
                    ],
                    "diagnostic_data_sha256": config["evaluation"][
                        "diagnostic"
                    ]["sha256"],
                    "decoding": config["evaluation"]["decoding"],
                    "dev_predictions_sha256": "a" * 64,
                    "challenge_predictions_sha256": "b" * 64,
                },
            )
            output = root / "candidate.json"
            candidate = build_sft_authorization_candidate(
                root, config_path, packet_path, output
            )
            self.assertEqual(
                candidate["activation_status"], "CANDIDATE_NOT_ACTIVE"
            )
            self.assertEqual(
                candidate["decision"], "AUTHORIZE_SFT_TRAINING"
            )
            self.assertTrue(candidate["human_approval_required"])
            self.assertFalse(
                (root / "reports/authorizations/bp2_sft.json").exists()
            )

    def test_incomplete_base_packet_cannot_unlock_sft(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, config = self.make_repo(root)
            packet_path = root / "base_packet.json"
            write_json(
                packet_path,
                {
                    "packet_version": "bp4-base-evaluation-v1",
                    "status": "PASS",
                    "target_contract_version": config["evaluation"][
                        "target_contract_version"
                    ],
                    "dev_rows": 330,
                    "challenge_rows": 140,
                    "human_gold_used": False,
                    "challenge_usage": "diagnostic_only",
                    "selection_data_sha256": config["evaluation"]["selection"][
                        "sha256"
                    ],
                    "diagnostic_data_sha256": config["evaluation"][
                        "diagnostic"
                    ]["sha256"],
                    "decoding": config["evaluation"]["decoding"],
                    "dev_predictions_sha256": "a" * 64,
                    "challenge_predictions_sha256": "b" * 64,
                },
            )
            with self.assertRaisesRegex(
                ValueError, "Base packet prerequisite failed: dev_rows"
            ):
                build_sft_authorization_candidate(
                    root, config_path, packet_path, root / "candidate.json"
                )

    def test_candidate_cannot_be_written_as_active_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, config = self.make_repo(root)
            packet_path = root / "base_packet.json"
            write_json(
                packet_path,
                {
                    "packet_version": "bp4-base-evaluation-v1",
                    "status": "PASS",
                    "target_contract_version": config["evaluation"][
                        "target_contract_version"
                    ],
                    "dev_rows": 331,
                    "challenge_rows": 140,
                    "human_gold_used": False,
                    "challenge_usage": "diagnostic_only",
                    "selection_data_sha256": config["evaluation"]["selection"][
                        "sha256"
                    ],
                    "diagnostic_data_sha256": config["evaluation"][
                        "diagnostic"
                    ]["sha256"],
                    "decoding": config["evaluation"]["decoding"],
                    "dev_predictions_sha256": "a" * 64,
                    "challenge_predictions_sha256": "b" * 64,
                },
            )
            active_path = root / "reports/authorizations/bp2_sft.json"
            with self.assertRaisesRegex(
                ValueError, "cannot be written to the active path"
            ):
                build_sft_authorization_candidate(
                    root, config_path, packet_path, active_path
                )
