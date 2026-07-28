import json
import unittest

from scripts.prepare_annotation_packet import ANNOTATION_COLUMNS
from scripts.validate_annotations import calculate_agreement, validate_rows


def expected(sample_id="sample-1", source_text="物流很快"):
    return {column: "" for column in ANNOTATION_COLUMNS} | {
        "sample_id": sample_id,
        "source_text": source_text,
    }


def completed(sample_id="sample-1", source_text="物流很快", keywords=None, annotator="A01"):
    return expected(sample_id, source_text) | {
        "annotator_id": annotator,
        "keywords_json": json.dumps(keywords or ["物流", "快"], ensure_ascii=False),
        "annotation_status": "complete",
    }


class AnnotationValidationTests(unittest.TestCase):
    def test_valid_completed_row_passes(self):
        issues, normalized = validate_rows([completed()], [expected()], "primary")
        self.assertEqual(issues, [])
        self.assertEqual(normalized["sample-1"]["keywords"], ["物流", "快"])

    def test_modified_source_is_rejected(self):
        issues, _ = validate_rows([completed(source_text="物流较慢")], [expected()], "primary")
        self.assertIn("modified", {item["code"] for item in issues})

    def test_non_source_and_too_long_keywords_are_rejected(self):
        row = completed(keywords=["配送", "物流非常快"])
        issues, _ = validate_rows([row], [expected()], "primary")
        self.assertIn("not_in_source", {item["code"] for item in issues})
        self.assertIn("invalid_keyword", {item["code"] for item in issues})

    def test_empty_keyword_list_requires_notes(self):
        row = completed(keywords=[])
        row["keywords_json"] = "[]"
        issues, _ = validate_rows([row], [expected()], "primary")
        self.assertIn("missing_for_empty", {item["code"] for item in issues})

    def test_excluded_row_requires_reason_and_no_keywords(self):
        row = completed()
        row["annotation_status"] = "exclude"
        issues, _ = validate_rows([row], [expected()], "primary")
        codes = {item["code"] for item in issues}
        self.assertIn("missing", codes)
        self.assertIn("not_empty_for_exclude", codes)

    def test_agreement_outputs_only_nonmatching_pairs(self):
        primary = {
            "a": {"sample_id": "a", "source_text": "物流很快", "annotator_id": "A01", "status": "complete", "keywords": ["物流", "快"], "exclude_reason": "", "notes": ""},
            "b": {"sample_id": "b", "source_text": "包装完整", "annotator_id": "A01", "status": "complete", "keywords": ["包装"], "exclude_reason": "", "notes": ""},
        }
        secondary = {
            "a": {**primary["a"], "annotator_id": "A02", "keywords": ["快", "物流"]},
            "b": {**primary["b"], "annotator_id": "A02", "keywords": ["完整"]},
        }
        metrics, disagreements = calculate_agreement(primary, secondary)
        self.assertEqual(metrics["auto_agreed_rows"], 1)
        self.assertEqual(metrics["disagreement_rows"], 1)
        self.assertEqual(disagreements[0]["sample_id"], "b")


if __name__ == "__main__":
    unittest.main()
