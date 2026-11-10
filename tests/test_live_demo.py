import json
import unittest

from prompts.prompt_template_3 import get_keyword_extraction_prompt_3
from demo.live.inference import inspect_response, parse_model_json, prepare_messages


class LiveDemoTests(unittest.TestCase):
    def test_input_limits_and_original_prompt(self):
        for source in [None, "", " \n", "字" * 513]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                prepare_messages(source)
        messages = prepare_messages("屏幕清晰。")
        self.assertEqual([item["role"] for item in messages], ["system", "user"])
        self.assertIn("屏幕清晰。", messages[1]["content"])
        system, user = get_keyword_extraction_prompt_3("屏幕清晰。")
        self.assertEqual(messages, [{"role": "system", "content": system}, {"role": "user", "content": user}])
        self.assertEqual(len(prepare_messages("字" * 512)), 2)

    def test_only_numeric_score_strings_are_normalized(self):
        payload = {"keywords": [["描述屏幕", "屏幕", "0.9"]]}
        raw = "思考：分析原文。\nJSON输出：\n```json\n" + json.dumps(payload) + "\n```"
        result = inspect_response("屏幕清晰。", raw)
        self.assertEqual(parse_model_json(raw), payload)
        self.assertEqual(result["payload"], {"keywords": [["描述屏幕", "屏幕", 0.9]]})
        self.assertIn("keywords[0].confidence_invalid", result["raw_errors"])
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["normalizations"], ["keywords[0].confidence_string_to_number"])
        self.assertEqual(result["rows"][0][1], "0.9")

    def test_bad_scores_and_other_errors_are_not_repaired(self):
        for score in ["NaN", "Infinity", "bad", True, "1.5"]:
            raw = json.dumps({"keywords": [["描述", "续航", score], ["描述", "续航", 0.9]]})
            with self.subTest(score=score):
                result = inspect_response("屏幕", raw)
                self.assertIn("keywords[0].confidence_invalid", result["errors"])
                self.assertIn("keywords[0].keyword_not_in_source", result["errors"])
                self.assertIn("keywords[1].keyword_duplicate", result["errors"])
                self.assertEqual([row[0] for row in result["rows"]], ["续航", "续航"])

    def test_source_errors_and_overlapping_matches(self):
        payload = {"keywords": [["描述", word, 0.9] for word in ["屏幕", "屏幕清晰", "屏幕", "续航"]]}
        source = "屏幕清晰，屏幕清晰。"
        result = inspect_response(source, json.dumps(payload))
        self.assertIn("keywords[2].keyword_duplicate", result["errors"])
        self.assertIn("keywords[3].keyword_not_in_source", result["errors"])
        self.assertEqual("".join(text for text, _ in result["highlights"]), source)
        self.assertEqual([text for text, label in result["highlights"] if label], ["屏幕清晰", "屏幕清晰"])
        self.assertEqual(result["rows"][-1][-1], "No")

    def test_bad_and_truncated_output_cannot_pass(self):
        for raw in ["", '{"keywords": [', '{"keywords": [["x", "屏幕", NaN]]}', '{"keywords": [["x", "屏幕", 1e999]]}']:
            with self.subTest(raw=raw):
                self.assertIsNone(parse_model_json(raw))
                self.assertTrue(inspect_response("屏幕", raw)["errors"])
        raw = json.dumps({"keywords": [["描述", "屏幕", 0.9]]})
        self.assertEqual(inspect_response("屏幕", raw)["errors"], [])
        self.assertTrue(inspect_response("屏幕", raw, completed=False)["errors"])


if __name__ == "__main__":
    unittest.main()
