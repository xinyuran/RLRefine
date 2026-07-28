import tempfile
import unittest
from pathlib import Path

from evaluation.second_schema_benchmark import (
    _validate_frozen_data,
    evaluate_records,
    parse_prediction,
    response_format,
)
from scripts.build_second_schema_assets import run as build_assets


class SecondSchemaBenchmarkTests(unittest.TestCase):
    def test_response_format_is_strict_closed_object(self):
        payload = response_format()
        schema = payload["json_schema"]["schema"]
        self.assertTrue(payload["json_schema"]["strict"])
        self.assertFalse(schema["additionalProperties"])

    def test_formal_benchmark_rejects_unreviewed_draft(self):
        with tempfile.TemporaryDirectory() as temporary:
            draft = Path(temporary) / "draft"
            build_assets(draft)
            with self.assertRaisesRegex(ValueError, "frozen manifest requires"):
                _validate_frozen_data(draft)

    def test_parser_separates_formal_schema_from_source_contract(self):
        source = "包裹还没送到，请尽快处理。"
        valid = parse_prediction("a", source, '{"intent":"delivery","urgency":"high","evidence":"包裹还没送到"}')
        unfaithful = parse_prediction("b", source, '{"intent":"delivery","urgency":"high","evidence":"已经签收"}')
        self.assertTrue(valid["schema_valid"])
        self.assertTrue(valid["target_contract_valid"])
        self.assertTrue(unfaithful["schema_valid"])
        self.assertFalse(unfaithful["target_contract_valid"])

    def test_metrics_cover_routing_contract_and_wall_clock_throughput(self):
        gold = [
            {"sample_id": "a", "source": "申请退款", "target": {"intent": "refund", "urgency": "normal", "evidence": "申请退款"}},
            {"sample_id": "b", "source": "包裹未到", "target": {"intent": "delivery", "urgency": "high", "evidence": "包裹未到"}},
        ]
        predictions = [
            {
                **parse_prediction("a", "申请退款", '{"intent":"refund","urgency":"normal","evidence":"申请退款"}'),
                "transport_error": None, "request_seconds": 1.0, "ttft_seconds": 0.1, "output_tokens": 40,
            },
            {
                **parse_prediction("b", "包裹未到", '{"intent":"delivery","urgency":"low","evidence":"包裹未到"}'),
                "transport_error": None, "request_seconds": 1.5, "ttft_seconds": 0.2, "output_tokens": 20,
            },
        ]
        report = evaluate_records(gold, predictions, wall_clock_seconds=2.0)
        self.assertEqual(report["urgency_accuracy"], 0.5)
        self.assertEqual(report["schema_valid_rate"], 1.0)
        self.assertEqual(report["target_contract_valid_rate"], 1.0)
        self.assertEqual(report["throughput_output_tokens_per_wall_second"], 30.0)
        self.assertEqual(report["transport_error_count"], 0)


if __name__ == "__main__":
    unittest.main()
