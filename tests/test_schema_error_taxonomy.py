import unittest

from evaluation.schema_error_taxonomy import analyze, classify_prediction


def row(sample_id, raw, status="error", **extra):
    return {"sample_id": sample_id, "status": status, "raw_response": raw, **extra}


class SchemaErrorTaxonomyTests(unittest.TestCase):
    def test_classifies_first_contract_failure(self):
        self.assertEqual(classify_prediction(row("a", "not json")), "json_not_parseable")
        self.assertEqual(classify_prediction(row("b", "[]")), "top_level_not_object")
        self.assertEqual(classify_prediction(row("c", '{"keywords":[["x","too-long",0.5]]}')), "keyword_length_or_type_invalid")
        self.assertEqual(classify_prediction(row("d", '{"keywords":[["x","好",true]]}')), "confidence_not_numeric")

    def test_analysis_is_exhaustive_and_enforces_expected_count(self):
        report, rows = analyze([
            row("a", "not json"),
            row("b", '{"keywords":[["x","好",0.5]]}', status="success"),
        ], expected_errors=1)
        self.assertEqual(report["schema_error_count"], 1)
        self.assertEqual(rows[0]["category"], "json_not_parseable")
        with self.assertRaisesRegex(ValueError, "Expected 2"):
            analyze([row("a", "not json")], expected_errors=2)


if __name__ == "__main__":
    unittest.main()
