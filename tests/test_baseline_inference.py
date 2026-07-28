import unittest

from evaluation.baseline_inference import (
    build_prompt,
    parse_completion,
    prepare_dev_gold,
    prepare_human_gold,
)


def dev_row(sample_id="dev-1", source_text="屏幕清晰"):
    return {
        "sample_id": sample_id,
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": f"请分析以下评论并提取关键词：\n\n【待处理评论】\n{source_text}\n\n请严格按照规则输出。",
            },
        ],
        "solution": '{"keywords":[["属性","屏幕",0.9],["描述","清晰",0.8]]}',
    }


class BaselineInferenceContractTests(unittest.TestCase):
    def test_human_gold_contract_is_preserved(self):
        rows = prepare_human_gold([{
            "sample_id": "gold-1",
            "normalized_text": "屏幕清晰",
            "gold": {"keywords": ["屏幕", "清晰"]},
            "dataset_version": "keyword-gold-v1",
            "split": "test_gold",
        }])
        self.assertEqual(rows[0]["gold"]["keywords"], ["屏幕", "清晰"])

    def test_human_gold_keyword_must_be_in_source(self):
        with self.assertRaisesRegex(ValueError, "source contract"):
            prepare_human_gold([{
                "sample_id": "gold-1",
                "normalized_text": "屏幕清晰",
                "gold": {"keywords": ["物流"]},
                "dataset_version": "keyword-gold-v1",
                "split": "test_gold",
            }])

    def test_dev_reference_is_prepared_from_solution(self):
        rows = prepare_dev_gold([dev_row()])
        self.assertEqual(rows[0]["normalized_text"], "屏幕清晰")
        self.assertEqual(rows[0]["gold"], {"keywords": ["屏幕", "清晰"]})
        self.assertEqual(rows[0]["label_origin"], "teacher_dev_reference_not_human_gold")

    def test_duplicate_dev_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            prepare_dev_gold([dev_row(), dev_row()])

    def test_prompt_variants_are_distinct_and_include_source(self):
        b0 = build_prompt("b0_simple", "屏幕清晰")
        b1 = build_prompt("b1_structured_v3", "屏幕清晰")
        b2 = build_prompt("b2_compact_v1", "屏幕清晰")
        b3 = build_prompt("b3_balanced_v1", "屏幕清晰")
        sft_v2 = build_prompt("sft_v2_json_only_protocol_v1", "屏幕清晰")
        self.assertNotEqual(b0, b1)
        self.assertNotEqual(b1, b2)
        self.assertNotEqual(b2, b3)
        self.assertIn("屏幕清晰", b0[1])
        self.assertIn("屏幕清晰", b1[1])
        self.assertIn("屏幕清晰", b2[1])
        self.assertIn("屏幕清晰", b3[1])
        self.assertLess(len("".join(b2)), len("".join(b1)) / 2)
        self.assertIn("只输出一个合法JSON对象", b2[0])
        self.assertIn("逐字出现在原文", b2[0])
        self.assertIn("最多15个", b2[0])
        self.assertLess(len("".join(b3)), len("".join(b1)) / 2)
        self.assertIn("连续原文子串", b3[0])
        self.assertIn("每项恰好三个值", b3[0])
        self.assertIn("检查：", b3[0])
        self.assertIn("屏幕清晰", sft_v2[1])
        self.assertIn("不要输出思考过程", sft_v2[0])
        self.assertIn("置信度必须是JSON number", sft_v2[0])
        self.assertNotIn("必须先进行明确的内部推理，再给出最终关键词", sft_v2[0])

    def test_valid_completion_is_standardized(self):
        row = parse_completion(
            "dev-1",
            "b0_simple",
            '前文示例是 {not-json}。\n最终答案：{"keywords":[["属性","屏幕",0.9]]}',
        )
        self.assertEqual(row["status"], "success")
        self.assertEqual(row["data"]["keywords"][0][1], "屏幕")

    def test_non_json_completion_is_explicit_error(self):
        row = parse_completion("dev-1", "b0_simple", "没有JSON")
        self.assertEqual(row["status"], "error")
        self.assertIsNone(row["data"])
        self.assertEqual(row["error_code"], "json_parse_failed")

    def test_schema_invalid_completion_is_explicit_error(self):
        row = parse_completion(
            "dev-1",
            "b0_simple",
            '{"keywords":[["属性","屏幕","0.9"]]}',
        )
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["error_code"], "schema_validation_failed")
        self.assertTrue(row["validation_errors"])


if __name__ == "__main__":
    unittest.main()
