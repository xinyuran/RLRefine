import unittest

from scripts.swift_yaml_launcher import build_launch


class SwiftYamlLauncherTests(unittest.TestCase):
    def test_yaml_is_expanded_to_explicit_cli_and_environment(self):
        command, env = build_launch({
            "ENV": {"CUDA_VISIBLE_DEVICES": "6,7", "NPROC_PER_NODE": "2"},
            "model": "/models/revision",
            "dataset": "data/train.jsonl",
            "val_dataset": "data/dev.jsonl",
            "load_best_model_at_end": True,
            "split_dataset_ratio": 0.0,
        })
        self.assertEqual(command[:2], ["swift", "sft"])
        self.assertIn("--model", command)
        self.assertEqual(command[command.index("--model") + 1], "/models/revision")
        self.assertEqual(command[command.index("--load_best_model_at_end") + 1], "true")
        self.assertNotIn("--ENV", command)
        self.assertEqual(env, {"CUDA_VISIBLE_DEVICES": "6,7", "NPROC_PER_NODE": "2"})

    def test_nested_non_environment_value_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Swift YAML value type"):
            build_launch({
                "ENV": {"CUDA_VISIBLE_DEVICES": "0"},
                "model": "model",
                "dataset": "train.jsonl",
                "val_dataset": "dev.jsonl",
                "model_kwargs": {"unsupported": True},
            })


if __name__ == "__main__":
    unittest.main()
