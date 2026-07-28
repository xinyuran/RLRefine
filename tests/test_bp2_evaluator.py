import unittest

from core.target_contract import build_messages
from evaluation.bp2_evaluator import evaluate_with_slices, prepare_reference


def asset(sample_id, source, keywords, slices):
    payload = {
        "keywords": [
            [f"原文包含{keyword}", keyword, 0.9] for keyword in keywords
        ]
    }
    return {
        "sample_id": sample_id,
        "split": "challenge",
        "label_origin": "teacher",
        "challenge_slices": slices,
        "messages": build_messages(source, payload),
    }


def prediction(sample_id, source_keyword, *, input_tokens=100, output_tokens=20):
    return {
        "sample_id": sample_id,
        "status": "success",
        "data": {"keywords": [["依据", source_keyword, 0.9]]},
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "request_seconds": 0.2,
    }


class Bp2EvaluatorTests(unittest.TestCase):
    def test_reference_preserves_slices_and_json_target_keywords(self):
        rows = [asset("a", "物流很快", ["物流", "很快"], ["negation"])]
        reference = prepare_reference(rows)
        self.assertEqual(reference[0]["gold"]["keywords"], ["物流", "很快"])
        self.assertEqual(reference[0]["challenge_slices"], ["negation"])

    def test_evaluation_reports_overall_slice_and_resource_metrics(self):
        reference = prepare_reference(
            [
                asset("a", "物流很快", ["物流"], ["long_context"]),
                asset("b", "包装很好", ["包装"], ["format_attack"]),
            ]
        )
        predictions = [
            prediction("a", "物流", input_tokens=100, output_tokens=10),
            prediction("b", "包装", input_tokens=200, output_tokens=20),
        ]
        report, errors = evaluate_with_slices(reference, predictions, "candidate")
        self.assertEqual(errors, [])
        self.assertEqual(report["macro"]["f1"], 1.0)
        self.assertEqual(report["challenge_slices"]["long_context"]["macro_f1"], 1.0)
        self.assertEqual(report["challenge_slices"]["format_attack"]["sample_count"], 1)
        self.assertEqual(report["resource_metrics"]["mean_input_tokens"], 150.0)
        self.assertIn("human_gold_forbidden", report["selection_policy"])


if __name__ == "__main__":
    unittest.main()
