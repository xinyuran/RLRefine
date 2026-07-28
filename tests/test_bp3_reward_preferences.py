import json
import tempfile
import unittest
from pathlib import Path

import yaml

from core.target_contract import build_messages
from evaluation.keyword_evaluator import file_sha256
from scripts.build_bp3_reward_preferences import (
    ATTACK_TYPES,
    build_package,
    load_config,
)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def parent_row(index, split):
    source = f"样品{index}物流快包装好"
    answer = {
        "keywords": [
            ["原文对象", "物流", 0.95],
            ["原文描述", "快", 0.9],
            ["原文对象", "包装", 0.85],
            ["原文描述", "好", 0.8],
        ]
    }
    return {
        "sample_id": f"{split}-{index}",
        "group_id": f"group-{split}-{index}",
        "split": split,
        "dataset_version": "keyword-v2.0.0",
        "target_contract_version": "keyword-json-target-v2",
        "messages": build_messages(source, answer),
    }


class Bp3RewardPreferenceTests(unittest.TestCase):
    def make_repo(self, root):
        root = Path(root)
        train = [parent_row(index, "train") for index in range(10)]
        dev = [parent_row(index, "dev") for index in range(5)]
        gold = [{"sample_id": f"gold-{index}"} for index in range(3)]
        manifest = root / "data/canonical/keyword_v2/manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text('{"status":"PASS"}', encoding="utf-8")
        train_path = root / "data/canonical/keyword_v2/train.jsonl"
        dev_path = root / "data/canonical/keyword_v2/dev.jsonl"
        gold_path = root / "data/canonical/keyword_v1/gold/v1/gold_test.jsonl"
        write_jsonl(train_path, train)
        write_jsonl(dev_path, dev)
        write_jsonl(gold_path, gold)

        template_path = Path(
            "configs/experiments/bp3_reward_preferences_v1.yaml"
        )
        config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        plugin_path = root / config["reward_v2"]["plugin"]
        plugin_path.parent.mkdir(parents=True, exist_ok=True)
        plugin_path.write_text("# synthetic test plugin fixture\n", encoding="utf-8")
        for name, path, count in (
            ("bp1_manifest", manifest, None),
            ("train", train_path, len(train)),
            ("dev", dev_path, len(dev)),
            ("forbidden_gold", gold_path, None),
        ):
            config["input"][name]["path"] = str(path.relative_to(root)).replace(
                "\\", "/"
            )
            config["input"][name]["sha256"] = file_sha256(path)
            if count is not None:
                config["input"][name]["rows"] = count
        config_path = root / "configs/experiments/bp3_test.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return config_path

    def test_config_freezes_exactly_fifty_attacks(self):
        config = load_config(
            Path("configs/experiments/bp3_reward_preferences_v1.yaml")
        )
        self.assertEqual(tuple(config["attacks"]["types"]), ATTACK_TYPES)
        self.assertEqual(config["attacks"]["case_count"], 50)
        self.assertEqual(config["attacks"]["per_type"], 5)

    def test_outputs_exactly_satisfy_bp2_dependencies(self):
        bp3 = load_config(
            Path("configs/experiments/bp3_reward_preferences_v1.yaml")
        )
        bp2 = yaml.safe_load(
            Path(
                "configs/experiments/bp2_post_training_pipeline_v1.yaml"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            bp3["output"]["dpo_train"], bp2["stages"]["dpo"]["dataset"]
        )
        self.assertEqual(
            bp3["output"]["dpo_dev"], bp2["stages"]["dpo"]["val_dataset"]
        )
        self.assertEqual(
            bp3["output"]["grpo_train"], bp2["stages"]["grpo"]["dataset"]
        )
        self.assertEqual(
            bp3["output"]["grpo_dev"], bp2["stages"]["grpo"]["val_dataset"]
        )
        self.assertIn(
            bp3["output"]["preference_gate"],
            bp2["stages"]["dpo"]["dependencies"],
        )
        self.assertIn(
            bp3["output"]["reward_gate"],
            bp2["stages"]["grpo"]["dependencies"],
        )
        self.assertEqual(
            bp3["reward_v2"]["plugin"],
            bp2["stages"]["grpo"]["args"]["external_plugins"],
        )

    def test_end_to_end_build_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self.make_repo(root)
            gate = build_package(root, config_path)
            self.assertEqual(
                gate["decision"], "ACCEPT_BP3_REWARD_AND_PREFERENCES"
            )
            self.assertFalse(gate["training_started"])
            report = root / "reports/bp3_keyword_v2"
            attack_rows = [
                json.loads(line)
                for line in (report / "reward_attack_cases.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(attack_rows), 50)
            self.assertTrue(all(row["margin"] > 0 for row in attack_rows))
            preference_gate = json.loads(
                (report / "preference_gate.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                preference_gate["counts"],
                {
                    "dpo_train": 10,
                    "dpo_dev": 5,
                    "grpo_train": 10,
                    "grpo_dev": 5,
                },
            )
            self.assertEqual(preference_gate["status"], "PASS")
            dpo_first = json.loads(
                (
                    root / "data/derived/keyword_v2/dpo_train.jsonl"
                ).read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(dpo_first["messages"][-1]["role"], "assistant")
            self.assertIn("rejected_response", dpo_first)
            self.assertNotIn("chosen", dpo_first)
            reward_gate = json.loads(
                (report / "reward_v2_gate.json").read_text(encoding="utf-8")
            )
            self.assertEqual(reward_gate["status"], "PASS")

    def test_gold_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self.make_repo(root)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            gold_path = root / config["input"]["forbidden_gold"]["path"]
            write_jsonl(gold_path, [{"sample_id": "train-0"}])
            config["input"]["forbidden_gold"]["sha256"] = file_sha256(gold_path)
            config_path.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Frozen human gold overlaps"):
                build_package(root, config_path)
