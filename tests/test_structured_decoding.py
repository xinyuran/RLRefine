import copy
import unittest

from evaluation.structured_decoding_diagnostics import analyze
from evaluation.structured_decoding_inference import (
    CONSTRAINED_VARIANT,
    CONTROL_VARIANT,
    build_request,
    keyword_response_format,
)


def prediction(sample_id, variant, keyword, status="success"):
    return {
        "sample_id": sample_id,
        "status": status,
        "data": {"keywords": [["属性", keyword, 0.9]]} if status == "success" else None,
        "error_code": None if status == "success" else "json_parse_failed",
        "prompt_variant": variant,
        "raw_response": "",
        "input_tokens": 100,
        "output_tokens": 20,
        "request_seconds": 0.1,
    }


class StructuredDecodingTests(unittest.TestCase):
    def test_response_schema_is_strict_and_excludes_optional_category(self):
        response_format = keyword_response_format()
        self.assertEqual(response_format["type"], "json_schema")
        schema = response_format["json_schema"]["schema"]
        self.assertEqual(schema["required"], ["keywords"])
        self.assertFalse(schema["additionalProperties"])
        self.assertNotIn("category", schema["properties"])
        item_schema = schema["properties"]["keywords"]["items"]
        self.assertEqual(item_schema["minItems"], 3)
        self.assertEqual(item_schema["maxItems"], 3)
        self.assertEqual(item_schema["prefixItems"][2]["type"], "number")

    def test_only_response_format_differs_between_arms(self):
        control = build_request("物流很快", "model", 512, 42, constrained=False)
        constrained = build_request("物流很快", "model", 512, 42, constrained=True)
        response_format = constrained.pop("response_format")
        self.assertEqual(control, constrained)
        self.assertEqual(response_format, keyword_response_format())

    def test_passing_candidate_is_adopted(self):
        reference = [
            {"sample_id": "a", "normalized_text": "物流很快", "gold": {"keywords": ["物流"]}},
            {"sample_id": "b", "normalized_text": "包装很好", "gold": {"keywords": ["包装"]}},
        ]
        frozen = [
            prediction("a", "b1_structured_v3", "物流"),
            prediction("b", "b1_structured_v3", "包装"),
        ]
        control = [
            prediction("a", CONTROL_VARIANT, "物流"),
            prediction("b", CONTROL_VARIANT, "包装"),
        ]
        constrained = [
            prediction("a", CONSTRAINED_VARIANT, "物流"),
            prediction("b", CONSTRAINED_VARIANT, "包装"),
        ]
        report, rows = analyze(
            reference, frozen, control, constrained,
            bootstrap_iterations=50, seed=42,
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(report["backend_fidelity_pass"])
        self.assertTrue(report["structured_decoding_pass"])
        self.assertEqual(report["decision"], "ADOPT_JSON_SCHEMA_DECODING")

    def test_backend_drift_invalidates_comparison(self):
        reference = [
            {"sample_id": "a", "normalized_text": "物流很快", "gold": {"keywords": ["物流"]}},
            {"sample_id": "b", "normalized_text": "包装很好", "gold": {"keywords": ["包装"]}},
        ]
        frozen = [
            prediction("a", "b1_structured_v3", "物流"),
            prediction("b", "b1_structured_v3", "包装"),
        ]
        control = [
            prediction("a", CONTROL_VARIANT, "很快"),
            prediction("b", CONTROL_VARIANT, "很好"),
        ]
        constrained = copy.deepcopy(control)
        for row in constrained:
            row["prompt_variant"] = CONSTRAINED_VARIANT
        report, _ = analyze(
            reference, frozen, control, constrained,
            bootstrap_iterations=50, seed=42,
        )
        self.assertFalse(report["backend_fidelity_pass"])
        self.assertEqual(report["decision"], "INVALID_BACKEND_COMPARISON")

    def test_valid_nonexact_prediction_is_not_a_runtime_error(self):
        reference = [
            {"sample_id": "a", "normalized_text": "物流很快", "gold": {"keywords": ["物流"]}},
        ]
        frozen = [prediction("a", "b1_structured_v3", "很快")]
        control = [prediction("a", CONTROL_VARIANT, "很快")]
        constrained = [prediction("a", CONSTRAINED_VARIANT, "很快")]
        report, _ = analyze(
            reference, frozen, control, constrained,
            bootstrap_iterations=20, seed=42,
        )
        metric = report["metrics"][CONSTRAINED_VARIANT]
        self.assertEqual(metric["runtime_error_count"], 0)
        self.assertEqual(metric["evaluation_difference_sample_count"], 1)
        self.assertTrue(
            report["structured_decoding_checks"]["constrained_no_runtime_errors"]
        )


if __name__ == "__main__":
    unittest.main()
