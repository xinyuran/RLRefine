import json
import logging
import unittest

from core.keyword_contract import MAX_KEYWORD_LENGTH as CORE_MAX_KEYWORD_LENGTH
from core.keyword_contract import MAX_KEYWORDS as CORE_MAX_KEYWORDS
from core.target_contract import build_messages
from rl.reward_builder_v2 import (
    MAX_KEYWORD_LENGTH,
    MAX_KEYWORDS,
    RewardV2Config,
    SchemaBasedRewardV2,
    parse_exact_json,
    schema_based_reward_v2,
    validate_structure,
)


def payload(*keywords, confidence=0.9):
    return {
        "keywords": [
            [f"原文包含{keyword}", keyword, confidence] for keyword in keywords
        ]
    }


def replace_keywords(reference, keywords):
    return {
        "keywords": [
            [f"原文包含{keyword}", keyword, 0.9] for keyword in keywords
        ]
    }


class RewardBuilderV2Tests(unittest.TestCase):
    def setUp(self):
        self.reward = SchemaBasedRewardV2()
        self.source = "物流速度很快，包装完好"
        self.reference = payload("物流", "速度", "很快", "包装", "完好")
        self.solution = json.dumps(self.reference, ensure_ascii=False)
        self.prompt = build_messages(self.source)

    def score(self, completion):
        return self.reward.score_with_details(
            completion, self.solution, self.prompt
        )

    def test_exact_json_is_required(self):
        self.assertIsNotNone(parse_exact_json(self.solution))
        self.assertIsNone(parse_exact_json(f"```json\n{self.solution}\n```"))
        self.assertEqual(self.score(f"```json\n{self.solution}\n```")["total"], -1.0)

    def test_structure_is_strict(self):
        extra = {**self.reference, "note": "attack"}
        self.assertEqual(validate_structure(extra), ["payload_top_level_invalid"])
        string_confidence = json.loads(self.solution)
        string_confidence["keywords"][0][2] = "0.9"
        string_score = self.score(json.dumps(string_confidence, ensure_ascii=False))
        self.assertFalse(string_score["schema_valid"])
        self.assertEqual(string_score["total"], -0.75)

    def test_perfect_solution_beats_hallucination_and_drop(self):
        perfect = self.score(self.solution)
        hallucinated = self.score(
            json.dumps(payload("物流", "速度", "很快", "包装", "完好", "不存在"), ensure_ascii=False)
        )
        dropped = self.score(
            json.dumps(replace_keywords(self.reference, ["物流", "速度"]), ensure_ascii=False)
        )
        self.assertGreater(perfect["total"], hallucinated["total"])
        self.assertGreater(perfect["total"], dropped["total"])
        self.assertGreater(hallucinated["hallucination_penalty"], 0)

    def test_atomicity_detects_merged_reference_keywords(self):
        merged = self.score(
            json.dumps(
                replace_keywords(self.reference, ["物流速度", "很快", "包装", "完好"]),
                ensure_ascii=False,
            )
        )
        self.assertLess(merged["atomicity"], 1.0)
        self.assertGreater(self.score(self.solution)["total"], merged["total"])

    def test_miscalibration_is_independently_visible(self):
        bad_payload = payload("物流", "速度", "很快", "包装", "完好", confidence=0.1)
        bad = self.score(json.dumps(bad_payload, ensure_ascii=False))
        good = self.score(self.solution)
        self.assertEqual(good["extraction_f1"], bad["extraction_f1"])
        self.assertLess(bad["count_calibration"], good["count_calibration"])
        self.assertGreater(good["total"], bad["total"])

    def test_weights_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            RewardV2Config(extraction_f1_weight=0.5)

    def test_ms_swift_messages_fallback_and_component_log(self):
        with self.assertLogs("rl.reward_builder_v2", logging.INFO) as captured:
            values = self.reward(
                [self.solution],
                [self.solution],
                prompts=[None],
                messages=[self.prompt],
            )
        self.assertEqual(len(values), 1)
        self.assertIn("reward_v2_component_summary", captured.output[0])

    def test_ms_swift_registry_export_is_a_reward_factory(self):
        self.assertIs(schema_based_reward_v2, SchemaBasedRewardV2)
        self.assertIsInstance(schema_based_reward_v2(), SchemaBasedRewardV2)

    def test_standalone_plugin_contract_matches_core_contract(self):
        self.assertEqual(MAX_KEYWORD_LENGTH, CORE_MAX_KEYWORD_LENGTH)
        self.assertEqual(MAX_KEYWORDS, CORE_MAX_KEYWORDS)
