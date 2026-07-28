import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.bp4_e4_grpo_from_dpo import _metric_summary, research_checks


class Bp4E4GrpoEvaluationTests(unittest.TestCase):
    def test_research_gate_accepts_preregistered_boundary(self):
        packet = {
            "sample_count": 331,
            "human_gold_used": False,
            "challenge_usage": "diagnostic_only",
            "baseline": {"micro_f1": 0.57, "schema_valid_rate": 0.74, "hallucination_keyword_rate": 0.024, "mean_output_tokens": 200, "max_token_output_count": 2},
            "candidate": {"micro_f1": 0.56, "schema_valid_rate": 0.80, "hallucination_keyword_rate": 0.024, "mean_output_tokens": 200, "max_token_output_count": 2},
            "paired": {"macro_f1_delta_candidate_minus_baseline": -0.01, "macro_f1_bootstrap_95_ci": [-0.039, 0.02]},
        }
        gates = {"macro_f1_delta_min": -0.01, "macro_f1_ci_low_gt": -0.04, "micro_f1_delta_min": -0.01, "schema_valid_rate_min": 0.80, "schema_valid_rate_delta_min": 0.03, "hallucination_keyword_rate_max": 0.03, "hallucination_keyword_rate_delta_max": 0.0, "mean_output_tokens_ratio_max": 1.0, "max_token_output_count_delta_max": 0}
        self.assertTrue(all(research_checks(packet, gates).values()))

    def test_evaluation_runner_never_launches_training(self):
        script = Path("scripts/run_bp4_e4_grpo_from_dpo_evaluation.sh").read_text(encoding="utf-8")
        self.assertNotIn("swift rlhf", script)
        self.assertNotIn("bp2_pipeline launch", script)
        self.assertIn('"training_started":false', script)

    def test_metric_adapter_supports_deployed_three_argument_helper(self):
        report = {"resource_metrics": {}, "micro": {"f1": 0.5}, "macro": {"f1": 0.5}, "schema_valid_rate": 1.0, "hallucination": {"keyword_rate": 0.0}, "error_sample_count": 0}
        telemetry = {"telemetry_version": "bp4-contract-aware-telemetry-v1"}
        with patch(
            "evaluation.bp4_e4_grpo_from_dpo.metric_summary",
            side_effect=lambda _r, _p, _t: {"micro_f1": 0.5},
        ):
            result = _metric_summary(report, [{"output_tokens": 768}], telemetry, 768)
        self.assertEqual(result["max_token_output_count"], 1)


if __name__ == "__main__":
    unittest.main()
