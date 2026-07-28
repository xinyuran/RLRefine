import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
import tests.test_bp4_base_v2 as base_fixture

from evaluation.bp4_dpo_reliability import (
    CHALLENGE_PREDICTIONS,
    DEV_PREDICTIONS,
    FEEDBACK,
    TWO_LEVEL_GATE,
    deployment_checks,
    package_evaluation,
    preflight,
    research_checks,
)
from evaluation.bp4_base_v2 import package_evaluation as package_base
from evaluation.bp4_stage_common import metric_summary
from evaluation.keyword_evaluator import file_sha256, write_jsonl_atomic


def metrics(**overrides):
    value = {
        "micro_f1": 0.53,
        "macro_f1": 0.49,
        "schema_valid_rate": 0.68,
        "hallucination_keyword_rate": 0.03,
        "mean_output_tokens": 150.0,
        "max_token_output_count": 6,
    }
    value.update(overrides)
    return value


def packet():
    return {
        "sample_count": 331,
        "human_gold_used": False,
        "challenge_usage": "diagnostic_only",
        "baseline": metrics(),
        "candidate": metrics(
            micro_f1=0.525,
            macro_f1=0.48,
            schema_valid_rate=0.82,
            hallucination_keyword_rate=0.02,
            mean_output_tokens=130.0,
            max_token_output_count=4,
        ),
        "paired": {
            "macro_f1_delta_candidate_minus_baseline": -0.01,
            "macro_f1_bootstrap_95_ci": [-0.04, 0.02],
        },
    }


def research_gate():
    return {
        "macro_f1_delta_min": -0.02,
        "macro_f1_ci_low_gt": -0.05,
        "micro_f1_delta_min": -0.01,
        "schema_valid_rate_min": 0.80,
        "schema_valid_rate_delta_min": 0.05,
        "hallucination_keyword_rate_max": 0.03,
        "hallucination_keyword_rate_delta_max": 0.0,
        "mean_output_tokens_ratio_max": 0.90,
        "max_token_output_count_delta_max": -1,
    }


class Bp4DpoReliabilityTests(unittest.TestCase):
    def test_metric_summary_counts_token_budget_not_batched_finish_reason(self):
        report = {
            "micro": {"f1": 0.5},
            "macro": {"f1": 0.5},
            "schema_valid_rate": 1.0,
            "hallucination": {"keyword_rate": 0.0},
            "resource_metrics": {
                "mean_input_tokens": 10.0,
                "mean_output_tokens": 20.0,
                "mean_request_seconds": 0.1,
            },
            "error_sample_count": 0,
        }
        telemetry = {
            "raw_hallucination_keyword_rate": 0.0,
            "telemetry_version": "bp4-contract-aware-telemetry-v1",
        }
        predictions = [
            {"finish_reason": "length", "output_tokens": 30},
            {"finish_reason": "stop", "output_tokens": 768},
        ]

        summary = metric_summary(
            report, predictions, telemetry, max_new_tokens=768
        )

        self.assertEqual(summary["max_token_output_count"], 1)

    def test_evaluation_runner_never_launches_training_or_grpo(self):
        script = Path(
            "scripts/run_bp4_dpo_reliability_evaluation.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("bp2_pipeline launch", script)
        self.assertNotIn("swift rlhf", script)
        self.assertIn('"training_started":false', script)
        self.assertIn('"grpo_authorized":false', script)

    def test_research_gate_accepts_preregistered_reliability_tradeoff(self):
        checks = research_checks(packet(), research_gate())
        self.assertTrue(all(checks.values()), checks)

    def test_research_gate_rejects_schema_gain_below_threshold(self):
        value = packet()
        value["candidate"]["schema_valid_rate"] = 0.72
        checks = research_checks(value, research_gate())
        self.assertFalse(checks["schema_valid_rate_min"])
        self.assertFalse(checks["schema_valid_rate_delta_min"])

    def test_deployment_gate_uses_base_output_length_not_sft_length(self):
        checks = deployment_checks(
            packet(),
            metrics(mean_output_tokens=120.0),
            {
                "schema_valid_rate_min": 0.80,
                "hallucination_keyword_rate_max": 0.03,
                "mean_output_tokens_vs_base_ratio_max": 1.0,
                "max_token_output_count": 4,
            },
        )
        self.assertFalse(checks["mean_output_tokens_vs_base_ratio_max"])

    def test_preflight_binds_sft_predictions_and_completed_dpo_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "training/checkpoint-74"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer_state.json").write_text(
                '{"global_step":74}', encoding="utf-8"
            )
            (checkpoint / "adapter_config.json").write_text(
                "{}", encoding="utf-8"
            )
            artifact = root / "artifacts.json"
            artifact.write_text(
                json.dumps({"best_model_checkpoint": str(checkpoint)}),
                encoding="utf-8",
            )
            sft_dev = (
                root
                / "reports/bp4_controlled_matrix/sft/dev_predictions.jsonl"
            )
            write_jsonl_atomic(
                sft_dev,
                [{"sample_id": "dev-1", "status": "success"}],
            )
            upstream = root / "reports/sft_packet.json"
            upstream.parent.mkdir(parents=True, exist_ok=True)
            upstream.write_text(
                json.dumps(
                    {
                        "candidate_checkpoint": str(
                            root / "training/sft-checkpoint-370"
                        ),
                        "candidate_predictions_sha256": file_sha256(sft_dev),
                    }
                ),
                encoding="utf-8",
            )
            reliability = {
                "schema_version": "structalign-bp4-reliability-repair-v1",
                "inputs": {
                    "paired_packet": "reports/sft_packet.json",
                    "expected": {
                        "paired_packet_sha256": file_sha256(upstream),
                    },
                },
                "dpo_repair_preregistration": {
                    "matrix_id": "E2_DPO_RELIABILITY",
                },
            }
            reliability_path = root / "reliability.yaml"
            reliability_path.write_text(
                yaml.safe_dump(reliability), encoding="utf-8"
            )
            matrix_path = root / "matrix.yaml"
            matrix_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version":
                            "structalign-bp4-controlled-matrix-v1",
                        "evaluation": {
                            "forbidden_selection": "frozen_human_gold"
                        },
                    }
                ),
                encoding="utf-8",
            )
            auth_path = root / "reports/auth.json"
            auth_path.parent.mkdir(parents=True, exist_ok=True)
            auth_path.write_text("{}", encoding="utf-8")
            bp2_path = root / "bp2.yaml"
            bp2_path.write_text(
                yaml.safe_dump(
                    {
                        "stages": {
                            "dpo": {
                                "authorization_file": "reports/auth.json"
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            authorization = {
                "activation_status": "ACTIVE_APPROVED",
                "approved_scope": "BP4_DPO_RELIABILITY_REPAIR_ONLY",
                "matrix_id": "E2_DPO_RELIABILITY",
                "model_path": json.loads(
                    upstream.read_text(encoding="utf-8")
                )["candidate_checkpoint"],
                "downstream_grpo_authorized": False,
            }
            with patch(
                "evaluation.bp4_dpo_reliability.read_authorization",
                return_value=(authorization, []),
            ):
                report = preflight(
                    root,
                    reliability_path,
                    matrix_path,
                    bp2_path,
                    artifact,
                )
            self.assertEqual(report["status"], "PASS")
            self.assertTrue(all(report["checks"].values()))

    def test_package_records_two_level_result_without_authorizing_grpo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = base_fixture.Bp4BaseV2Tests()
            matrix_path, task_path, _, task = helper.make_repo(root)
            package_base(root, matrix_path, task_path)
            base_dir = root / "reports/bp4_controlled_matrix/base_v2"
            sft_dir = root / "reports/bp4_controlled_matrix/sft"
            sft_dir.mkdir(parents=True)
            write_jsonl_atomic(
                sft_dir / "dev_predictions.jsonl",
                json.loads(
                    json.dumps(
                        [
                            json.loads(line)
                            for line in (
                                base_dir / "dev_predictions.jsonl"
                            ).read_text(encoding="utf-8").splitlines()
                            if line
                        ]
                    )
                ),
            )
            write_jsonl_atomic(
                root / DEV_PREDICTIONS,
                [
                    json.loads(line)
                    for line in (
                        base_dir / "dev_predictions.jsonl"
                    ).read_text(encoding="utf-8").splitlines()
                    if line
                ],
            )
            write_jsonl_atomic(
                root / CHALLENGE_PREDICTIONS,
                [
                    json.loads(line)
                    for line in (
                        base_dir / "challenge_predictions.jsonl"
                    ).read_text(encoding="utf-8").splitlines()
                    if line
                ],
            )
            checkpoint = root / "training/checkpoint-74"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer_state.json").write_text(
                '{"global_step":74}', encoding="utf-8"
            )
            (checkpoint / "adapter_config.json").write_text(
                "{}", encoding="utf-8"
            )
            artifacts = root / "artifacts.json"
            artifacts.write_text(
                json.dumps({"best_model_checkpoint": str(checkpoint)}),
                encoding="utf-8",
            )
            sft_packet = root / "reports/sft_packet.json"
            sft_packet.write_text(
                json.dumps(
                    {
                        "candidate_checkpoint": str(
                            root / "training/sft-checkpoint-370"
                        ),
                        "baseline": {"mean_output_tokens": 100.0},
                    }
                ),
                encoding="utf-8",
            )
            reliability = {
                "schema_version": "structalign-bp4-reliability-repair-v1",
                "inputs": {
                    "paired_packet": "reports/sft_packet.json",
                    "expected": {
                        "paired_packet_sha256": file_sha256(sft_packet),
                    },
                },
                "dpo_repair_preregistration": {
                    "matrix_id": "E2_DPO_RELIABILITY",
                    "bootstrap_iterations": 10000,
                    "bootstrap_seed": 42,
                    "decoding": {"max_new_tokens": 768},
                    "research_gate": research_gate(),
                    "deployment_gate": {
                        "schema_valid_rate_min": 0.90,
                        "hallucination_keyword_rate_max": 0.03,
                        "mean_output_tokens_vs_base_ratio_max": 1.0,
                        "max_token_output_count": 0,
                    },
                },
            }
            reliability_path = root / "reliability.yaml"
            reliability_path.write_text(
                yaml.safe_dump(reliability), encoding="utf-8"
            )
            bp2_path = root / "bp2.yaml"
            bp2_path.write_text("{}", encoding="utf-8")
            preflight_report = {
                "status": "PASS",
                "candidate_checkpoint": str(checkpoint),
                "authorization_sha256": "a" * 64,
            }
            evaluation_dir = (
                root / "reports/bp4_reliability_repair/dpo/evaluation"
            )
            evaluation_dir.mkdir(parents=True, exist_ok=True)
            for name, payload in (
                ("preflight.json", preflight_report),
                ("inference_manifest.json", {"status": "PASS"}),
                ("inference_runtime.json", {"status": "PASS"}),
            ):
                (evaluation_dir / name).write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            with (
                patch(
                    "evaluation.bp4_dpo_reliability.preflight",
                    return_value=preflight_report,
                ),
                patch(
                    "evaluation.bp4_dpo_reliability._source_lineage",
                    return_value={},
                ),
                patch(
                    "evaluation.bp4_dpo_reliability._git_lineage",
                    return_value={},
                ),
                patch(
                    "evaluation.bp4_dpo_reliability._environment_lineage",
                    return_value={},
                ),
            ):
                gate = package_evaluation(
                    root,
                    reliability_path,
                    matrix_path,
                    bp2_path,
                    artifacts,
                )
            self.assertEqual(gate["status"], "PASS")
            self.assertFalse(gate["grpo_training_authorized"])
            self.assertEqual(
                gate["deployment_decision"], "REJECT_DPO_DEPLOYMENT"
            )
            self.assertIn(
                "research_progression_passed",
                gate["deployment_failed_checks"],
            )
            self.assertTrue((root / TWO_LEVEL_GATE).is_file())
            feedback = json.loads(
                (root / FEEDBACK).read_text(encoding="utf-8")
            )
            self.assertFalse(feedback["grpo_training_authorized"])


if __name__ == "__main__":
    unittest.main()
