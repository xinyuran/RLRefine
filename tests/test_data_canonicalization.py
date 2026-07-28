import json
import unittest

from scripts.canonicalize_sft_data import canonicalize_record


def build_record(keyword="清晰", score=0.9, comment="屏幕清晰"):
    response = json.dumps(
        {"keywords": [["评价", keyword, score]]},
        ensure_ascii=False,
    )
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": f"【待处理评论】\n{comment}\n\n请严格输出。",
            },
            {"role": "assistant", "content": f"<think>分析原文关键词。</think>\n{response}"},
        ]
    }


class DataCanonicalizationTests(unittest.TestCase):
    def test_numeric_string_confidence_is_safely_converted(self):
        canonical, reasons, converted, _ = canonicalize_record(
            build_record(score="0.90"),
            1,
            set(),
        )

        self.assertEqual(reasons, [])
        self.assertEqual(converted, 1)
        assistant = canonical["messages"][2]["content"]
        parsed = json.loads(assistant[assistant.find("{"):assistant.rfind("}") + 1])
        self.assertEqual(parsed["keywords"][0][2], 0.9)
        self.assertIsInstance(parsed["keywords"][0][2], float)

    def test_keyword_over_four_characters_is_quarantined(self):
        canonical, reasons, _, _ = canonicalize_record(
            build_record(keyword="屏幕非常清晰", comment="屏幕非常清晰"),
            1,
            set(),
        )

        self.assertIsNone(canonical)
        self.assertIn("keyword_too_long", reasons)

    def test_keyword_not_in_source_is_quarantined(self):
        canonical, reasons, _, _ = canonicalize_record(
            build_record(keyword="发热"),
            1,
            set(),
        )

        self.assertIsNone(canonical)
        self.assertIn("keyword_not_in_source", reasons)

    def test_second_duplicate_input_is_quarantined(self):
        seen = set()
        first, first_reasons, _, _ = canonicalize_record(build_record(), 1, seen)
        second, second_reasons, _, _ = canonicalize_record(build_record(), 2, seen)

        self.assertIsNotNone(first)
        self.assertEqual(first_reasons, [])
        self.assertIsNone(second)
        self.assertIn("duplicate_input", second_reasons)

    def test_empty_keywords_are_quarantined(self):
        record = build_record()
        record["messages"][2]["content"] = '{"keywords": []}'

        canonical, reasons, _, _ = canonicalize_record(record, 1, set())

        self.assertIsNone(canonical)
        self.assertIn("keywords_empty", reasons)

    def test_out_of_range_confidence_is_quarantined(self):
        canonical, reasons, _, _ = canonicalize_record(
            build_record(score=1.5),
            1,
            set(),
        )

        self.assertIsNone(canonical)
        self.assertIn("confidence_out_of_range", reasons)


if __name__ == "__main__":
    unittest.main()
