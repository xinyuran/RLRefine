import unittest

from evaluation.bp4_acceptance import authorization_candidate, decide_stage


def metrics(**overrides):
    value = {
        "micro_f1": 0.60,
        "macro_f1": 0.60,
        "schema_valid_rate": 0.95,
        "hallucination_keyword_rate": 0.02,
        "mean_output_tokens": 100.0,
        "max_token_output_count": 0,
    }
    value.update(overrides)
    return value


def packet():
    return {
        "packet_version": "bp4-paired-evaluation-v1",
        "selection_split": "teacher_dev",
        "sample_count": 331,
        "human_gold_used": False,
        "challenge_usage": "diagnostic_only",
        "target_contract_version": "keyword-json-target-v2",
        "bootstrap_iterations": 10000,
        "bootstrap_seed": 42,
        "selection_data_sha256": "a" * 64,
        "baseline_predictions_sha256": "b" * 64,
        "candidate_predictions_sha256": "c" * 64,
        "baseline_model_id": "base",
        "candidate_model_id": "candidate",
        "baseline_checkpoint": "base-snapshot",
        "candidate_checkpoint": "checkpoint-500",
        "baseline_stage": "base",
        "baseline": metrics(
            micro_f1=0.55,
            macro_f1=0.55,
            schema_valid_rate=0.90,
            hallucination_keyword_rate=0.025,
            mean_output_tokens=120.0,
        ),
        "candidate": metrics(),
        "paired": {
            "macro_f1_delta_candidate_minus_baseline": 0.05,
            "macro_f1_bootstrap_95_ci": [0.03, 0.07],
        },
    }


def gates():
    return {
        "paired_macro_f1_delta_min": 0.02,
        "paired_macro_f1_ci_low_gt": 0.0,
        "micro_f1_delta_min": 0.0,
        "schema_valid_rate_min": 0.90,
        "schema_valid_rate_delta_min": 0.0,
        "hallucination_keyword_rate_max": 0.03,
        "hallucination_keyword_rate_delta_max": 0.0,
        "mean_output_tokens_ratio_max": 1.0,
        "max_token_output_count_delta_max": 0,
        "require_complete_dev_coverage": True,
        "challenge_is_diagnostic_only": True,
    }


class Bp4AcceptanceTests(unittest.TestCase):
    def test_passing_sft_candidate_is_accepted(self):
        report = decide_stage(
            packet(), gates(), stage="sft", candidate_checkpoint="checkpoint-500"
        )
        self.assertEqual(report["decision"], "ACCEPT_SFT_CANDIDATE")
        self.assertTrue(all(report["checks"].values()))

    def test_single_regression_rejects_candidate(self):
        value = packet()
        value["candidate"]["hallucination_keyword_rate"] = 0.04
        report = decide_stage(
            value, gates(), stage="sft", candidate_checkpoint="checkpoint-500"
        )
        self.assertEqual(report["decision"], "REJECT_SFT_CANDIDATE")
        self.assertIn("hallucination_keyword_rate_max", report["failed_checks"])

    def test_stage_must_compare_against_its_direct_upstream(self):
        value = packet()
        value["baseline_stage"] = "base"
        with self.assertRaisesRegex(ValueError, "Invalid direct upstream"):
            decide_stage(
                value,
                gates(),
                stage="dpo",
                candidate_checkpoint="checkpoint-500",
            )

    def test_checkpoint_must_match_evaluation_packet(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            decide_stage(
                packet(),
                gates(),
                stage="sft",
                candidate_checkpoint="checkpoint-other",
            )

    def test_human_gold_packet_is_rejected_before_decision(self):
        value = packet()
        value["human_gold_used"] = True
        with self.assertRaisesRegex(ValueError, "Human gold"):
            decide_stage(
                value, gates(), stage="dpo", candidate_checkpoint="checkpoint-100"
            )

    def test_incomplete_dev_coverage_is_rejected(self):
        value = packet()
        value["sample_count"] = 330
        with self.assertRaisesRegex(ValueError, "exactly 331"):
            decide_stage(
                value, gates(), stage="grpo", candidate_checkpoint="checkpoint-100"
            )

    def test_rejected_upstream_cannot_authorize_downstream(self):
        rejected = decide_stage(
            {
                **packet(),
                "candidate": metrics(hallucination_keyword_rate=0.04),
            },
            gates(),
            stage="sft",
            candidate_checkpoint="checkpoint-500",
        )
        with self.assertRaisesRegex(ValueError, "rejected upstream"):
            authorization_candidate(
                rejected,
                next_stage="dpo",
                bp2_config_sha256="config",
                model_path="checkpoint-500",
            )

    def test_accepted_upstream_produces_inactive_bound_candidate(self):
        accepted = decide_stage(
            packet(), gates(), stage="sft", candidate_checkpoint="checkpoint-500"
        )
        candidate = authorization_candidate(
            accepted,
            next_stage="dpo",
            bp2_config_sha256="config-hash",
            model_path="checkpoint-500",
            dataset_sha256="train-hash",
            val_dataset_sha256="dev-hash",
        )
        self.assertEqual(candidate["decision"], "AUTHORIZE_DPO_TRAINING")
        self.assertEqual(candidate["activation_status"], "CANDIDATE_NOT_ACTIVE")
        self.assertTrue(candidate["human_approval_required"])
        self.assertEqual(candidate["dataset_sha256"], "train-hash")
