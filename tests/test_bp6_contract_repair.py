import unittest

from evaluation.bp6_contract_repair import repair_payload, repair_prediction


class Bp6ContractRepairTests(unittest.TestCase):
    def test_removes_only_invalid_duplicate_and_non_source_items(self):
        raw = '{"keywords":[["ok","物流",0.9],["dup","物流",0.8],["bad","虚构",0.7],["long","配送速度慢",0.6]]}'
        payload, actions = repair_payload(raw, "物流很快")
        self.assertEqual(payload, {"keywords": [["ok", "物流", 0.9]]})
        self.assertEqual(actions, ["drop_keyword_duplicate", "drop_keyword_not_in_source", "drop_keyword_length_or_type_invalid"])

    def test_never_fabricates_when_no_recoverable_item_exists(self):
        row = repair_prediction({"sample_id": "a", "status": "error", "raw_response": "not json"}, "物流很快")
        self.assertEqual(row["status"], "error")
        self.assertIsNone(row["data"])
        self.assertIn("json_not_parseable", row["repair_actions"])


if __name__ == "__main__":
    unittest.main()
