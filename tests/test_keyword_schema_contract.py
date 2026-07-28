import unittest

from core.config import Config
from core.keyword_contract import MAX_KEYWORD_LENGTH, MAX_KEYWORDS
from examples.keyword_extraction.schema import create_keyword_schema
from rl.reward_builder import RewardBuilder


class KeywordSchemaContractTests(unittest.TestCase):
    def setUp(self):
        self.schema = create_keyword_schema()

    def test_valid_keyword_tuples_pass(self):
        valid, errors = self.schema.validate({"keywords": [["属性", "屏幕", 0.9]]})
        self.assertTrue(valid, errors)

    def test_keyword_item_must_have_exactly_three_values(self):
        valid, errors = self.schema.validate({"keywords": [["属性", "屏幕"]]})
        self.assertFalse(valid)
        self.assertTrue(any("exactly 3" in error for error in errors))

    def test_keyword_item_is_not_an_object(self):
        valid, _ = self.schema.validate(
            {"keywords": [{"category": "属性", "keyword": "屏幕", "confidence": 0.9}]}
        )
        self.assertFalse(valid)

    def test_keyword_length_limit_is_four(self):
        valid, _ = self.schema.validate({"keywords": [["属性", "配送速度慢", 0.9]]})
        self.assertFalse(valid)

    def test_confidence_must_be_numeric_and_not_boolean(self):
        for confidence in ("0.9", True):
            with self.subTest(confidence=confidence):
                valid, _ = self.schema.validate({"keywords": [["属性", "屏幕", confidence]]})
                self.assertFalse(valid)

    def test_confidence_range_is_enforced(self):
        valid, _ = self.schema.validate({"keywords": [["属性", "屏幕", 1.1]]})
        self.assertFalse(valid)

    def test_keyword_count_limit_is_shared(self):
        config = Config()
        reward = RewardBuilder.create_keyword_reward()
        too_many = [["属性", "屏幕", 0.9] for _ in range(MAX_KEYWORDS + 1)]
        valid, _ = self.schema.validate({"keywords": too_many})
        self.assertFalse(valid)
        self.assertEqual(config.post_process_max_keyword_length, MAX_KEYWORD_LENGTH)
        self.assertEqual(reward.config.max_keyword_length, MAX_KEYWORD_LENGTH)
        self.assertEqual(reward.config.max_items, MAX_KEYWORDS)

    def test_json_schema_describes_tuple_items(self):
        schema = self.schema.to_json_schema()
        keywords = schema["properties"]["keywords"]
        self.assertEqual(keywords["type"], "array")
        self.assertEqual(keywords["maxItems"], MAX_KEYWORDS)
        self.assertEqual(keywords["items"]["maxItems"], 3)
        self.assertEqual(keywords["items"]["prefixItems"][2]["type"], "number")
        reward_keywords = RewardBuilder.create_keyword_reward().schema["properties"]["keywords"]
        self.assertEqual(reward_keywords["maxItems"], MAX_KEYWORDS)
        self.assertEqual(reward_keywords["items"]["prefixItems"][1]["maxLength"], MAX_KEYWORD_LENGTH)


if __name__ == "__main__":
    unittest.main()
