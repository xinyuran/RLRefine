import json
import unittest

from scripts.freeze_gold_test import build_gold_rows, validate_adjudications


def disagreement(sample_id="sample-1", source="物流很快"):
    return {
        "sample_id": sample_id,
        "source_text": source,
        "primary_annotator_id": "A01",
        "primary_status": "complete",
        "primary_keywords_json": json.dumps(["物流"], ensure_ascii=False),
        "primary_exclude_reason": "",
        "primary_notes": "",
        "secondary_annotator_id": "A02",
        "secondary_status": "complete",
        "secondary_keywords_json": json.dumps(["物流", "快"], ensure_ascii=False),
        "secondary_exclude_reason": "",
        "secondary_notes": "",
        "adjudicator_id": "",
        "adjudicated_status": "",
        "adjudicated_keywords_json": "",
        "adjudication_reason": "",
        "adjudication_notes": "",
    }


def adjudicated(**overrides):
    row = disagreement()
    row.update({
        "adjudicator_id": "A03",
        "adjudicated_status": "complete",
        "adjudicated_keywords_json": json.dumps(["物流", "快"], ensure_ascii=False),
        "adjudication_reason": "secondary_preferred",
        "adjudication_notes": "保留明确评价词",
    })
    row.update(overrides)
    return row


class GoldFreezeTests(unittest.TestCase):
    def test_valid_adjudication_passes(self):
        issues, normalized = validate_adjudications([adjudicated()], [disagreement()], ["A01", "A02"])
        self.assertEqual(issues, [])
        self.assertEqual(normalized["sample-1"]["keywords"], ["物流", "快"])

    def test_modified_frozen_column_is_rejected(self):
        issues, _ = validate_adjudications([adjudicated(source_text="物流较慢")], [disagreement()], ["A01", "A02"])
        self.assertIn("modified", {item["code"] for item in issues})

    def test_adjudicator_must_be_independent(self):
        issues, _ = validate_adjudications([adjudicated(adjudicator_id="A01")], [disagreement()], ["A01", "A02"])
        self.assertIn("not_independent", {item["code"] for item in issues})

    def test_preferred_reason_must_match_selected_label(self):
        issues, _ = validate_adjudications(
            [adjudicated(adjudicated_keywords_json=json.dumps(["物流"], ensure_ascii=False))],
            [disagreement()],
            ["A01", "A02"],
        )
        self.assertIn("reason_result_mismatch", {item["code"] for item in issues})

    def test_invalid_keyword_is_rejected(self):
        issues, _ = validate_adjudications(
            [adjudicated(adjudicated_keywords_json=json.dumps(["配送速度慢"], ensure_ascii=False), adjudication_reason="new_decision")],
            [disagreement()],
            ["A01", "A02"],
        )
        codes = {item["code"] for item in issues}
        self.assertIn("not_in_source", codes)
        self.assertIn("invalid_keyword", codes)

    def test_build_gold_combines_primary_agreed_and_adjudicated(self):
        candidates = [
            {"sample_id": "a", "group_id": "a", "source_line": 1},
            {"sample_id": "b", "group_id": "b", "source_line": 2},
            {"sample_id": "c", "group_id": "c", "source_line": 3},
        ]
        primary = {
            key: {"sample_id": key, "source_text": "物流很快", "annotator_id": "A01", "status": "complete", "keywords": ["物流"], "exclude_reason": "", "notes": ""}
            for key in "abc"
        }
        secondary = {
            "b": {**primary["b"], "annotator_id": "A02"},
            "c": {**primary["c"], "annotator_id": "A02", "keywords": ["快"]},
        }
        resolved = {
            "c": {"sample_id": "c", "source_text": "物流很快", "adjudicator_id": "A03", "status": "complete", "keywords": ["物流", "快"], "reason": "merged", "notes": "合并"}
        }
        gold, excluded, counts = build_gold_rows(candidates, primary, secondary, resolved)
        self.assertEqual(len(gold), 3)
        self.assertEqual(excluded, [])
        self.assertEqual(counts["human_primary_single"], 1)
        self.assertEqual(counts["human_double_agreed"], 1)
        self.assertEqual(counts["human_adjudicated"], 1)


if __name__ == "__main__":
    unittest.main()
