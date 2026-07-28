import json
import unittest

from rl.convert_sft_to_grpo import convert_single_sample
from rl.reward_builder import RewardBuilder


class SchemaBasedRewardTests(unittest.TestCase):
    def setUp(self):
        self.reward = RewardBuilder.create_keyword_reward()
        self.prompt = [
            {"role": "system", "content": "关键词抽取"},
            {
                "role": "user",
                "content": "【待处理评论】\n屏幕清晰，电池耐用。\n\n请严格输出。",
            },
        ]
        self.solution = json.dumps(
            {
                "keywords": [
                    ["屏幕属性", "清晰", 0.9],
                    ["电池属性", "耐用", 0.8],
                ]
            },
            ensure_ascii=False,
        )

    def test_perfect_answer_has_full_f1_and_no_hallucination(self):
        details = self.reward.score_with_details(
            self.solution,
            self.solution,
            self.prompt,
        )

        self.assertTrue(details["valid_json"])
        self.assertTrue(details["schema_valid"])
        self.assertEqual(details["f1"], 1.0)
        self.assertEqual(details["hallucination_penalty"], 0.0)

    def test_string_confidence_fails_schema_gate(self):
        completion = '{"keywords": [["屏幕属性", "清晰", "0.9"]]}'

        details = self.reward.score_with_details(
            completion,
            self.solution,
            self.prompt,
        )

        self.assertFalse(details["schema_valid"])
        self.assertEqual(details["accuracy"], 0.0)

    def test_quality_component_never_exceeds_configured_weight(self):
        parsed = {
            "keywords": [
                ["屏幕属性", "清晰", 0.9],
                ["电池属性", "耐用", 0.8],
            ]
        }

        self.assertLessEqual(
            self.reward._evaluate_quality(parsed),
            self.reward.config.quality_weight,
        )

    def test_hallucination_reduces_total_reward(self):
        faithful = '{"keywords": [["屏幕属性", "清晰", 0.9]]}'
        hallucinated = '{"keywords": [["屏幕属性", "发热", 0.9]]}'

        faithful_details = self.reward.score_with_details(
            faithful,
            self.solution,
            self.prompt,
        )
        hallucinated_details = self.reward.score_with_details(
            hallucinated,
            self.solution,
            self.prompt,
        )

        self.assertGreater(
            hallucinated_details["hallucination_penalty"],
            faithful_details["hallucination_penalty"],
        )
        self.assertLess(hallucinated_details["total"], faithful_details["total"])

    def test_missing_solution_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "solution"):
            self.reward([self.solution], prompts=[self.prompt])

    def test_component_summary_is_logged(self):
        self.reward.config.log_every_n_calls = 1

        with self.assertLogs("rl.reward_builder", level="INFO") as captured:
            self.reward(
                [self.solution],
                [self.solution],
                prompts=[self.prompt],
            )

        self.assertTrue(any("REWARD_METRICS" in line for line in captured.output))

    def test_input_contract_summary_is_logged_without_text(self):
        with self.assertLogs("rl.reward_builder", level="INFO") as captured:
            self.reward(
                [self.solution],
                [self.solution],
                prompts=[self.prompt],
                sample_id=["sample-001"],
            )

        contract_lines = [line for line in captured.output if "REWARD_INPUT_CONTRACT" in line]
        self.assertEqual(len(contract_lines), 1)
        self.assertIn('"solution_source": "positional"', contract_lines[0])
        self.assertIn('"prompt_source": "prompts_kwarg"', contract_lines[0])
        self.assertIn('"kwarg_keys": ["prompts", "sample_id"]', contract_lines[0])
        self.assertNotIn("屏幕清晰", contract_lines[0])

    def test_messages_kwarg_is_used_when_prompts_are_none(self):
        hallucinated = '{"keywords": [["屏幕属性", "发热", 0.9]]}'

        with self.assertLogs("rl.reward_builder", level="INFO") as captured:
            rewards = self.reward(
                [hallucinated],
                [self.solution],
                prompts=[None],
                messages=[self.prompt],
            )

        expected = self.reward.score_with_details(
            hallucinated,
            self.solution,
            self.prompt,
        )
        contract_lines = [line for line in captured.output if "REWARD_INPUT_CONTRACT" in line]
        self.assertEqual(rewards, [expected["total"]])
        self.assertGreater(expected["hallucination_penalty"], 0.0)
        self.assertIn('"prompt_source": "messages_kwarg"', contract_lines[0])
        self.assertIn('"first_item_type": "list"', contract_lines[0])
        self.assertNotIn("屏幕清晰", contract_lines[0])


class GrpoConversionTests(unittest.TestCase):
    def test_converter_preserves_assistant_as_solution(self):
        sample = {
            "sample_id": "sample-001",
            "source_line": 7,
            "contract_version": "keyword-v1",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
                {"role": "assistant", "content": "gold"},
            ]
        }

        converted = convert_single_sample(sample, True, "think")

        self.assertEqual(converted["solution"], "gold")
        self.assertEqual(converted["sample_id"], "sample-001")
        self.assertEqual(converted["source_line"], 7)
        self.assertEqual(converted["contract_version"], "keyword-v1")
        self.assertEqual(
            [message["role"] for message in converted["messages"]],
            ["system", "user"],
        )

    def test_converter_rejects_sample_without_reference(self):
        sample = {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ]
        }

        self.assertIsNone(convert_single_sample(sample, True, "think"))


if __name__ == "__main__":
    unittest.main()
