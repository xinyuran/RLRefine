import unittest
from unittest.mock import patch

from evaluation.bp5_serving_benchmark import _metrics


class Bp5ServingTelemetryTests(unittest.TestCase):
    @patch("evaluation.bp5_serving_benchmark.evaluate_records")
    def test_metrics_separate_contract_errors_and_use_wall_clock_throughput(self, evaluate_records):
        evaluate_records.return_value = (
            {
                "macro": {"f1": 0.5},
                "micro": {"f1": 0.6},
                "schema_valid_rate": 0.5,
                "hallucination": {"keyword_rate": 0.1},
            },
            [],
        )
        predictions = [
            {
                "status": "success",
                "request_seconds": 2.0,
                "ttft_seconds": 0.5,
                "output_tokens": 100,
            },
            {
                "status": "error",
                "request_seconds": 1.0,
                "ttft_seconds": 0.2,
                "output_tokens": 50,
            },
        ]

        metrics = _metrics([], predictions, wall_clock_seconds=2.5)

        self.assertEqual(metrics["target_contract_error_count"], 1)
        self.assertEqual(metrics["transport_error_count"], 0)
        self.assertEqual(metrics["throughput_output_tokens_per_wall_second"], 60.0)
        self.assertEqual(metrics["wall_clock_seconds"], 2.5)
        self.assertNotIn("runtime_error_count", metrics)
        self.assertNotIn("throughput_output_tokens_per_second", metrics)


if __name__ == "__main__":
    unittest.main()
