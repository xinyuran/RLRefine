import argparse
import json
import tempfile
import unittest
from pathlib import Path

from core.target_contract import (
    DATASET_VERSION,
    TARGET_CONTRACT_VERSION,
    build_messages,
    validate_keyword_payload,
)
from evaluation.keyword_evaluator import file_sha256
from scripts.build_bp1_assets import (
    FORMAT_ATTACK_SUFFIX,
    build_assets,
    build_format_attacks,
    build_review_queue,
    classify_slices,
    normalize_keyword_payload,
    read_jsonl,
    select_natural_challenge,
)
from scripts.validate_bp1_assets import build_feedback, build_gate_report


def payload(keyword="物流"):
    return {"keywords": [["原文直接提到该对象", keyword, 0.9]]}


def parent(index, *, source=None, keyword="物流", split="train"):
    source = source or (
        f"第{index}条评论但是物流没有按时送到，真是绝了，"
        + "包装和商品描述很多，" * 12
    )
    return {
        "sample_id": f"sample-{index:04d}",
        "group_id": f"group-{index:04d}",
        "split": split,
        "contract_version": "keyword-v1",
        "source_line": index + 1,
        "messages": build_messages(source, payload(keyword)),
    }


def dense_parent(index, *, split="train"):
    keywords = [
        "物流",
        "没有",
        "送到",
        "包装",
        "商品",
        "描述",
        "评论",
        "按时",
        "真是",
        "绝了",
        "但是",
        "按时送到",
    ]
    source = (
        f"第{index}条评论，但是物流没有按时送到，真是绝了，"
        "包装和商品描述很多。" + "评论商品包装物流很多，" * 16
    )
    value = {
        "keywords": [
            [f"原文包含关键词{keyword}", keyword, 0.9 - position * 0.01]
            for position, keyword in enumerate(keywords)
        ]
    }
    return {
        "sample_id": f"sample-{index:04d}",
        "group_id": f"group-{index:04d}",
        "split": split,
        "contract_version": "keyword-v1",
        "source_line": index + 1,
        "messages": build_messages(source, value),
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class TargetContractTests(unittest.TestCase):
    def test_json_only_messages_share_one_versioned_contract(self):
        value = payload()
        messages = build_messages("物流很快", value)
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[-1]["content"], json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        self.assertNotIn("<think>", messages[-1]["content"])
        self.assertTrue(validate_keyword_payload(value, "物流很快")[0])
        self.assertEqual(TARGET_CONTRACT_VERSION, "keyword-json-target-v2")
        self.assertEqual(DATASET_VERSION, "keyword-v2.0.0")

    def test_payload_rejects_hallucinated_and_duplicate_keywords(self):
        value = {
            "keywords": [
                ["依据", "物流", 0.9],
                ["重复", "物流", 0.8],
                ["原文没有", "客服", 0.7],
            ]
        }
        valid, errors = validate_keyword_payload(value, "物流很快")
        self.assertFalse(valid)
        self.assertIn("keywords[1].keyword_duplicate", errors)
        self.assertIn("keywords[2].keyword_not_in_source", errors)


class Bp1BuilderTests(unittest.TestCase):
    def test_slice_classification_covers_natural_difficulties(self):
        source = "但是物流没有送到，我也是醉了。" + "描述" * 80
        value = {"keywords": [["依据", "物流", 0.9]] * 12}
        slices = classify_slices(source, value)
        self.assertIn("negation", slices)
        self.assertIn("irony", slices)
        self.assertIn("ambiguity", slices)
        self.assertIn("long_context", slices)
        self.assertIn("long_keyword_list", slices)

    def test_duplicate_keyword_repair_preserves_first_occurrence(self):
        value = {
            "keywords": [
                ["first", "物流", 0.9],
                ["duplicate", "物流", 0.8],
                ["other", "很快", 0.7],
            ]
        }
        normalized, removed = normalize_keyword_payload(value)
        self.assertEqual(removed, 1)
        self.assertEqual(
            normalized["keywords"],
            [["first", "物流", 0.9], ["other", "很快", 0.7]],
        )

    def test_challenge_selection_is_deterministic_and_removed_from_train(self):
        rows = [dense_parent(index) for index in range(140)]
        first, first_remaining = select_natural_challenge(rows, 100)
        second, second_remaining = select_natural_challenge(list(reversed(rows)), 100)
        self.assertEqual(
            [row["sample_id"] for row, _ in first],
            [row["sample_id"] for row, _ in second],
        )
        selected = {row["sample_id"] for row, _ in first}
        self.assertFalse(selected & {row["sample_id"] for row in first_remaining})
        self.assertEqual(len(first_remaining), 40)
        self.assertEqual(
            {row["sample_id"] for row in first_remaining},
            {row["sample_id"] for row in second_remaining},
        )

    def test_format_attack_is_declared_and_keeps_parent_label(self):
        selected = [(parent(1), ["negation"])]
        attack = build_format_attacks(selected, 1)[0]
        self.assertIn("format_attack", attack["challenge_slices"])
        self.assertEqual(attack["parent_sample_id"], "sample-0001")
        source = attack["messages"][1]["content"]
        self.assertIn(FORMAT_ATTACK_SUFFIX.strip(), source)
        self.assertEqual(json.loads(attack["messages"][2]["content"]), payload())

    def test_quarantine_review_queue_is_not_evaluation_gold(self):
        quarantine = [
            {
                "sample_id": f"q-{index}",
                "source_line": index,
                "reasons": ["keyword_not_in_source"],
                "record": parent(index)["messages"] and {
                    "messages": parent(index)["messages"]
                },
            }
            for index in range(5)
        ]
        queue = build_review_queue(quarantine, set(), 5)
        self.assertEqual(len(queue), 5)
        self.assertTrue(all("gold" not in row for row in queue))
        self.assertTrue(
            all(row["label_status"].startswith("needs_independent") for row in queue)
        )

    def test_end_to_end_build_gate_and_feedback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_file = root / "train.jsonl"
            dev_file = root / "dev.jsonl"
            quarantine_file = root / "quarantine.jsonl"
            gold_file = root / "gold.jsonl"
            output_dir = root / "keyword_v2"
            report_dir = root / "reports"
            train = [dense_parent(index) for index in range(130)]
            dev = [
                dense_parent(1000 + index, split="dev")
                for index in range(5)
            ]
            quarantine = []
            for index in range(60):
                row = parent(2000 + index)
                quarantine.append(
                    {
                        "sample_id": f"q-{index}",
                        "source_line": index,
                        "reasons": ["keyword_not_in_source"],
                        "record": {"messages": row["messages"]},
                    }
                )
            gold = [
                {
                    "sample_id": "gold-only",
                    "dataset_version": "keyword-gold-v1",
                    "split": "test_gold",
                }
            ]
            write_jsonl(train_file, train)
            write_jsonl(dev_file, dev)
            write_jsonl(quarantine_file, quarantine)
            write_jsonl(gold_file, gold)
            args = argparse.Namespace(
                train_file=train_file,
                dev_file=dev_file,
                quarantine_file=quarantine_file,
                gold_file=gold_file,
                expected_gold_sha256=file_sha256(gold_file),
                output_dir=output_dir,
                report_dir=report_dir,
                natural_challenge_size=100,
                format_attack_count=10,
                review_queue_size=50,
            )
            manifest = build_assets(args)
            self.assertEqual(manifest["counts"]["train"], 30)
            self.assertEqual(manifest["counts"]["challenge"], 110)
            self.assertEqual(len(read_jsonl(output_dir / "dev.jsonl")), 5)
            gate = build_gate_report(
                output_dir,
                report_dir,
                file_sha256(train_file),
                file_sha256(dev_file),
                file_sha256(gold_file),
                expected_parent_train_rows=130,
                expected_dev_rows=5,
            )
            self.assertEqual(gate["status"], "PASS")
            feedback = build_feedback(gate)
            self.assertEqual(
                feedback["next_work_package"], "BP2_POST_TRAINING_PIPELINE"
            )
            self.assertIn("expensive_training_still_requires", feedback["authorization"])


if __name__ == "__main__":
    unittest.main()
