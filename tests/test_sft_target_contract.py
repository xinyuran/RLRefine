import unittest

from scripts.audit_sft_target_contract import audit_rows, summarize


def row(assistant: str):
    return {
        "messages": [
            {"role": "system", "content": "training-system"},
            {"role": "user", "content": "【待处理评论】\n屏幕清晰\n\n请严格输出。"},
            {"role": "assistant", "content": assistant},
        ]
    }


class SftTargetContractTests(unittest.TestCase):
    def test_long_reasoning_before_json_is_counted(self):
        report = audit_rows(
            [row("<think>很长推理</think>" + '{"keywords":[]}')],
            token_count=len,
            max_new_tokens=10,
        )
        self.assertEqual(report["counts"]["think_tag_targets"], 1)
        self.assertEqual(report["counts"]["json_starts_at_or_after_max_new_tokens"], 1)
        self.assertEqual(report["counts"]["json_only_over_max_new_tokens"], 1)

    def test_missing_json_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing target JSON"):
            audit_rows([row("<think>only</think>")], token_count=len)

    def test_summary_is_deterministic(self):
        self.assertEqual(summarize([1, 2, 3])["p50"], 2)


if __name__ == "__main__":
    unittest.main()
