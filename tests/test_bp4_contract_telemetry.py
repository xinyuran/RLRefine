import unittest

from evaluation.bp4_contract_telemetry import contract_aware_telemetry


class Bp4ContractTelemetryTests(unittest.TestCase):
    def test_invalid_contract_output_still_counts_hallucination(self):
        predictions = [
            {
                "sample_id": "a",
                "status": "error",
                "error_code": "target_contract_failed",
                "validation_errors": ["keywords[0].keyword_not_in_source"],
                "raw_response": '{"keywords":[["说明","虚构词",0.9]]}',
            },
            {
                "sample_id": "b",
                "status": "success",
                "raw_response": '{"keywords":[["说明","物流",0.9]]}',
            },
        ]
        report = contract_aware_telemetry(
            predictions, {"a": "物流很好", "b": "物流很快"}
        )
        self.assertEqual(report["raw_keyword_count"], 2)
        self.assertEqual(report["raw_hallucinated_keyword_count"], 1)
        self.assertEqual(report["raw_hallucination_keyword_rate"], 0.5)
        self.assertEqual(
            report["contract_error_counts"]["keyword_not_in_source"], 1
        )

    def test_malformed_json_is_reported_without_fake_keyword_denominator(self):
        report = contract_aware_telemetry(
            [
                {
                    "sample_id": "a",
                    "status": "error",
                    "raw_response": "not json",
                }
            ],
            {"a": "物流很好"},
        )
        self.assertEqual(report["exact_json_parseable_rate"], 0.0)
        self.assertEqual(report["raw_keyword_count"], 0)
        self.assertEqual(
            report["contract_error_counts"]["exact_json_parse_failed"], 1
        )


if __name__ == "__main__":
    unittest.main()
