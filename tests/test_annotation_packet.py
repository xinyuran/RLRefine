import json
import math
import unittest

from scripts.prepare_annotation_packet import ANNOTATION_COLUMNS, prepare_rows


def candidate(index):
    keyword = f"词{index}"
    response = json.dumps(
        {"keywords": [["评价", keyword, 0.9]]},
        ensure_ascii=False,
    )
    return {
        "sample_id": f"sample-{index:03d}",
        "group_id": f"group-{index:03d}",
        "source_line": index + 1,
        "split": "test_candidate",
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": f"【待处理评论】\n商品{keyword}\n\n请严格输出。",
            },
            {"role": "assistant", "content": response},
        ],
    }


class AnnotationPacketTests(unittest.TestCase):
    def test_primary_packet_is_blind_and_covers_all_candidates(self):
        primary, _, _ = prepare_rows([candidate(0), candidate(1)])

        self.assertEqual(len(primary), 2)
        self.assertEqual(tuple(primary[0]), ANNOTATION_COLUMNS)
        self.assertEqual(primary[0]["keywords_json"], "")
        self.assertNotIn("teacher_keywords", primary[0])

    def test_secondary_packet_uses_at_least_requested_ratio(self):
        source = [candidate(index) for index in range(11)]
        _, secondary, _ = prepare_rows(source, secondary_ratio=0.20)

        self.assertEqual(len(secondary), math.ceil(11 * 0.20))

    def test_packet_is_deterministic_and_input_order_independent(self):
        source = [candidate(index) for index in range(10)]

        first = prepare_rows(source, seed="seed")
        second = prepare_rows(list(reversed(source)), seed="seed")

        self.assertEqual(first, second)

    def test_teacher_reference_is_kept_separate(self):
        _, _, teacher = prepare_rows([candidate(3)])

        self.assertEqual(teacher[0]["teacher_keywords"], ["词3"])
        self.assertNotIn("source_text", teacher[0])

    def test_non_test_candidate_is_rejected(self):
        record = candidate(0)
        record["split"] = "train"

        with self.assertRaisesRegex(ValueError, "test_candidate"):
            prepare_rows([record])


if __name__ == "__main__":
    unittest.main()
