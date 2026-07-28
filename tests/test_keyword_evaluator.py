import unittest

from evaluation.keyword_evaluator import evaluate_records


def gold(sample_id, text="屏幕清晰", keywords=None):
    return {
        "sample_id": sample_id,
        "normalized_text": text,
        "gold": {"keywords": keywords or ["屏幕", "清晰"]},
    }


def prediction(sample_id, keywords=None, status="success"):
    data = None if status == "error" else {
        "keywords": [["属性", keyword, 0.9] for keyword in (keywords or ["屏幕", "清晰"])]
    }
    return {"sample_id": sample_id, "status": status, "data": data}


class KeywordEvaluatorTests(unittest.TestCase):
    def test_perfect_predictions_score_one(self):
        report, errors = evaluate_records([gold("a")], [prediction("a")], "perfect")
        self.assertEqual(report["micro"]["f1"], 1.0)
        self.assertEqual(report["macro"]["f1"], 1.0)
        self.assertEqual(report["exact_set_match_rate"], 1.0)
        self.assertEqual(errors, [])

    def test_partial_prediction_has_expected_metrics(self):
        report, _ = evaluate_records(
            [gold("a", keywords=["屏幕", "清晰"])],
            [prediction("a", keywords=["屏幕", "发热"])],
            "partial",
        )
        self.assertEqual(report["micro"]["precision"], 0.5)
        self.assertEqual(report["micro"]["recall"], 0.5)
        self.assertEqual(report["micro"]["f1"], 0.5)

    def test_hallucination_is_measured_against_source(self):
        report, errors = evaluate_records(
            [gold("a", text="屏幕清晰", keywords=["屏幕"])],
            [prediction("a", keywords=["屏幕", "发热"])],
            "hallucinated",
        )
        self.assertEqual(report["hallucination"]["keyword_rate"], 0.5)
        self.assertEqual(errors[0]["hallucinated"], ["发热"])

    def test_error_status_counts_as_zero_task_score(self):
        report, errors = evaluate_records([gold("a")], [prediction("a", status="error")], "error")
        self.assertEqual(report["micro"]["f1"], 0.0)
        self.assertEqual(report["schema_valid_rate"], 0.0)
        self.assertEqual(report["status_counts"], {"error": 1})
        self.assertEqual(len(errors), 1)

    def test_invalid_success_data_is_schema_invalid(self):
        invalid = {"sample_id": "a", "status": "success", "data": {"keywords": ["屏幕"]}}
        report, errors = evaluate_records([gold("a")], [invalid], "invalid")
        self.assertEqual(report["schema_valid_rate"], 0.0)
        self.assertTrue(errors[0]["schema_errors"])

    def test_prediction_coverage_must_match_gold(self):
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            evaluate_records([gold("a")], [], "missing")

    def test_duplicate_sample_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate sample_id"):
            evaluate_records([gold("a")], [prediction("a"), prediction("a")], "duplicate")


if __name__ == "__main__":
    unittest.main()
