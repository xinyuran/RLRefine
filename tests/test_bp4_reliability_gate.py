import unittest
from evaluation.bp4_reliability_gate import (
    authorization_candidate,
    build_amendment,
)


def metrics(**overrides):
    value = {
        "micro_f1": 0.50,
        "macro_f1": 0.50,
        "schema_valid_rate": 0.70,
        "hallucination_keyword_rate": 0.03,
        "mean_output_tokens": 160.0,
        "max_token_output_count": 6,
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
        "candidate_model_id": "sft",
        "baseline_checkpoint": "base",
        "candidate_checkpoint": "checkpoint-370",
        "baseline_stage": "base",
        "baseline": metrics(
            micro_f1=0.28,
            macro_f1=0.25,
            schema_valid_rate=0.45,
            hallucination_keyword_rate=0.12,
            mean_output_tokens=106.0,
            max_token_output_count=0,
        ),
        "candidate": metrics(),
        "paired": {
            "macro_f1_delta_candidate_minus_baseline": 0.25,
            "macro_f1_bootstrap_95_ci": [0.20, 0.29],
        },
    }


def config():
    return {
        "research_progression_gate": {
            "paired_macro_f1_delta_min": 0.02,
            "paired_macro_f1_ci_low_gt": 0.0,
            "micro_f1_delta_min": 0.0,
            "schema_valid_rate_delta_min": 0.0,
            "hallucination_keyword_rate_delta_max": 0.0,
            "require_complete_dev_coverage": True,
            "challenge_is_diagnostic_only": True,
        },
        "deployment_gate": {
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
        },
        "dpo_repair_preregistration": {
            "matrix_id": "E2_DPO_RELIABILITY",
            "checkpoint_selection": "minimum_teacher_dev_eval_loss",
        },
    }


class Bp4ReliabilityGateTests(unittest.TestCase):
    def test_sft_can_pass_research_and_fail_deployment(self):
        value = build_amendment(
            packet(),
            {"decision": "REJECT_SFT_CANDIDATE"},
            {
                "decision": "REJECT_SFT_CANDIDATE",
                "acceptance_sha256": "acceptance",
                "paired_packet_sha256": "packet",
            },
            config(),
            packet_sha256="packet",
            legacy_acceptance_sha256="acceptance",
            legacy_gate_sha256="gate",
        )
        self.assertEqual(
            value["research_progression_decision"],
            "ACCEPT_SFT_RESEARCH_PROGRESSION",
        )
        self.assertEqual(value["deployment_decision"], "REJECT_SFT_DEPLOYMENT")
        self.assertEqual(value["legacy_decision"], "REJECT_SFT_CANDIDATE")

    def test_legacy_hash_reference_must_match(self):
        with self.assertRaisesRegex(ValueError, "acceptance"):
            build_amendment(
                packet(),
                {"decision": "REJECT_SFT_CANDIDATE"},
                {
                    "decision": "REJECT_SFT_CANDIDATE",
                    "acceptance_sha256": "wrong",
                    "paired_packet_sha256": "packet",
                },
                config(),
                packet_sha256="packet",
                legacy_acceptance_sha256="acceptance",
                legacy_gate_sha256="gate",
            )

    def test_authorization_is_inactive_and_dpo_only(self):
        amendment = build_amendment(
            packet(),
            {"decision": "REJECT_SFT_CANDIDATE"},
            {
                "decision": "REJECT_SFT_CANDIDATE",
                "acceptance_sha256": "acceptance",
                "paired_packet_sha256": "packet",
            },
            config(),
            packet_sha256="packet",
            legacy_acceptance_sha256="acceptance",
            legacy_gate_sha256="gate",
        )
        candidate = authorization_candidate(amendment, config())
        self.assertEqual(candidate["activation_status"], "CANDIDATE_NOT_ACTIVE")
        self.assertEqual(candidate["stage"], "dpo")
        self.assertFalse(candidate["downstream_grpo_authorized"])
        self.assertTrue(candidate["human_approval_required"])
