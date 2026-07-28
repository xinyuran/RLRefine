import unittest

from evaluation.baseline_diagnostics import analyze


def reference(sample_id="a", text="屏幕清晰", keywords=None):
    return {
        "sample_id": sample_id,
        "normalized_text": text,
        "gold": {"keywords": keywords or ["屏幕", "清晰"]},
    }


def prediction(sample_id="a", keywords=None, variant="b0_simple", output_tokens=20):
    return {
        "sample_id": sample_id,
        "status": "success",
        "data": {"keywords": [["属性", keyword, 0.9] for keyword in (keywords or ["屏幕"])]},
        "prompt_variant": variant,
        "input_tokens": 100,
        "output_tokens": output_tokens,
        "raw_response": "{}",
    }


class BaselineDiagnosticsTests(unittest.TestCase):
    def test_paired_b1_win_is_measured(self):
        report, rows = analyze(
            [reference()],
            [prediction()],
            [prediction(keywords=["屏幕", "清晰"], variant="b1_structured_v3")],
            max_new_tokens=512,
            bootstrap_iterations=20,
            seed=42,
        )
        self.assertEqual(report["paired"]["outcomes"], {"b1_win": 1})
        self.assertGreater(report["paired"]["bootstrap_95_ci"][0], 0)
        self.assertGreater(rows[0]["f1_delta_b1_minus_b0"], 0)

    def test_schema_error_is_categorized(self):
        invalid = prediction()
        invalid.update({
            "status": "error",
            "data": None,
            "error_code": "schema_validation_failed",
            "validation_errors": ["confidence should be numeric type"],
        })
        report, _ = analyze(
            [reference()], [invalid],
            [prediction(variant="b1_structured_v3")],
            512, 10, 42,
        )
        self.assertEqual(
            report["variants"]["b0_simple"]["schema_issue_counts"]["confidence_not_numeric"],
            1,
        )

    def test_max_token_output_is_counted(self):
        report, _ = analyze(
            [reference()],
            [prediction(output_tokens=512)],
            [prediction(variant="b1_structured_v3")],
            512, 10, 42,
        )
        self.assertEqual(report["variants"]["b0_simple"]["max_token_output_count"], 1)

    def test_coverage_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            analyze([reference()], [], [], 512, 10, 42)

    def test_bootstrap_iterations_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "iterations must be positive"):
            analyze(
                [reference()],
                [prediction()],
                [prediction(variant="b1_structured_v3")],
                512, 0, 42,
            )


if __name__ == "__main__":
    unittest.main()
