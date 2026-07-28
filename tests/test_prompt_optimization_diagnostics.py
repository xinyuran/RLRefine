import unittest

from evaluation.prompt_optimization_diagnostics import analyze


def reference(sample_id="a", keywords=None):
    return {
        "sample_id": sample_id,
        "normalized_text": "屏幕清晰",
        "gold": {"keywords": keywords if keywords is not None else ["屏幕", "清晰"]},
    }


def prediction(
    variant,
    sample_id="a",
    keywords=None,
    input_tokens=100,
    output_tokens=50,
):
    values = keywords if keywords is not None else ["屏幕", "清晰"]
    return {
        "sample_id": sample_id,
        "status": "success",
        "data": {"keywords": [["依据", keyword, 0.9] for keyword in values]},
        "prompt_variant": variant,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "raw_response": "{}",
    }


class PromptOptimizationDiagnosticsTests(unittest.TestCase):
    def test_compact_noninferior_b2_is_selected(self):
        report, rows = analyze(
            [reference()],
            [prediction("b1_structured_v3", input_tokens=900, output_tokens=250)],
            [prediction("b2_compact_v1", input_tokens=300, output_tokens=80)],
            512, 20, 42,
        )
        self.assertEqual(report["provisional_decision"], "freeze_b2_prompt")
        self.assertEqual(report["paired"]["outcomes"], {"tie": 1})
        self.assertLess(report["token_cost_ratio_b2_over_b1"]["input"], 0.5)
        self.assertEqual(rows[0]["f1_delta_b2_minus_b1"], 0)

    def test_quality_drop_beyond_margin_rejects_b2(self):
        report, _ = analyze(
            [reference()],
            [prediction("b1_structured_v3", input_tokens=900, output_tokens=250)],
            [prediction("b2_compact_v1", keywords=["屏幕"], input_tokens=300, output_tokens=40)],
            512, 20, 42,
        )
        self.assertFalse(report["checks"]["b2_macro_f1_noninferior"])
        self.assertEqual(report["provisional_decision"], "retain_b1_or_iterate_b2")

    def test_b2_max_token_output_is_rejected(self):
        report, _ = analyze(
            [reference()],
            [prediction("b1_structured_v3", input_tokens=900, output_tokens=512)],
            [prediction("b2_compact_v1", input_tokens=300, output_tokens=512)],
            512, 20, 42,
        )
        self.assertFalse(report["checks"]["b2_no_max_token_outputs"])

    def test_coverage_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            analyze([reference()], [], [], 512, 20, 42)

    def test_invalid_noninferiority_margin_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "noninferiority margin"):
            analyze(
                [reference()],
                [prediction("b1_structured_v3")],
                [prediction("b2_compact_v1")],
                512, 20, 42, 1.0,
            )

    def test_b3_candidate_uses_dynamic_report_names(self):
        report, rows = analyze(
            [reference()],
            [prediction("b1_structured_v3", input_tokens=900, output_tokens=250)],
            [prediction("b3_balanced_v1", input_tokens=350, output_tokens=100)],
            512, 20, 42, 0.03, "b3_balanced_v1",
        )
        self.assertEqual(report["provisional_decision"], "freeze_b3_prompt")
        self.assertIn("token_cost_ratio_b3_over_b1", report)
        self.assertIn("f1_delta_b3_minus_b1", rows[0])

    def test_sft_v2_protocol_keeps_b1_input_budget(self):
        report, _ = analyze(
            [reference()],
            [prediction("b1_structured_v3", input_tokens=900, output_tokens=250)],
            [prediction("sft_v2_json_only_protocol_v1", input_tokens=920, output_tokens=100)],
            512, 20, 42, 0.03, "sft_v2_json_only_protocol_v1",
        )
        self.assertTrue(report["checks"]["sft_input_tokens_within_five_percent"])
        self.assertEqual(report["provisional_decision"], "freeze_sft_prompt")


if __name__ == "__main__":
    unittest.main()
