import unittest

from evaluation.sft_acceptance import decide


def reference(sample_id="a", keywords=None):
    return {
        "sample_id": sample_id,
        "normalized_text": "屏幕清晰",
        "gold": {"keywords": keywords or ["屏幕", "清晰"]},
    }


def prediction(keywords, sample_id="a", output_tokens=20):
    return {
        "sample_id": sample_id,
        "status": "success",
        "data": {"keywords": [["属性", keyword, 0.9] for keyword in keywords]},
        "input_tokens": 100,
        "output_tokens": output_tokens,
    }


def gates():
    return {
        "paired_macro_f1_delta_min": 0.02,
        "paired_macro_f1_bootstrap_95_ci_low_gt": 0.0,
        "micro_f1_min": 0.5,
        "schema_valid_rate_min": 0.9,
        "hallucination_keyword_rate_max": 0.03,
        "mean_output_tokens_max": 30,
        "max_token_output_count_max": 0,
    }


class SftAcceptanceTests(unittest.TestCase):
    def test_candidate_that_passes_every_preregistered_gate_is_accepted(self):
        report, _ = decide(
            [reference()],
            [prediction(["屏幕"])],
            [prediction(["屏幕", "清晰"])],
            gates(),
            bootstrap_iterations=20,
        )
        self.assertEqual(report["decision"], "ACCEPT")
        self.assertTrue(all(report["checks"].values()))

    def test_single_failed_gate_rejects_candidate(self):
        strict = gates()
        strict["mean_output_tokens_max"] = 10
        report, _ = decide(
            [reference()],
            [prediction(["屏幕"])],
            [prediction(["屏幕", "清晰"], output_tokens=20)],
            strict,
            bootstrap_iterations=20,
        )
        self.assertEqual(report["decision"], "REJECT")
        self.assertFalse(report["checks"]["mean_output_tokens_max"])


if __name__ == "__main__":
    unittest.main()
