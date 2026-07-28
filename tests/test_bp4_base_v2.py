import json
import tempfile
import unittest
from pathlib import Path

import yaml

from core.target_contract import (
    DATASET_VERSION,
    TARGET_CONTRACT_VERSION,
    build_messages,
)
from evaluation.bp4_base_v2 import (
    finish_reason,
    package_evaluation,
    parse_completion,
    prepare_inference_rows,
)
from evaluation.keyword_evaluator import (
    file_sha256,
    write_json_atomic,
    write_jsonl_atomic,
)
from scripts.bp4_matrix import base_task


def asset(index, split):
    source = f"样品{index}物流很好"
    payload = {"keywords": [["原文对象", "物流", 0.95]]}
    return {
        "sample_id": f"{split}-{index}",
        "group_id": f"group-{split}-{index}",
        "split": split,
        "dataset_version": DATASET_VERSION,
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "label_origin": "teacher_reference",
        "challenge_slices": ["long_context"] if split == "challenge" else [],
        "messages": build_messages(source, payload),
    }


def prediction(row):
    return {
        "sample_id": row["sample_id"],
        "status": "success",
        "data": {"keywords": [["原文对象", "物流", 0.95]]},
        "raw_response": '{"keywords":[["原文对象","物流",0.95]]}',
        "input_tokens": 100,
        "output_tokens": 20,
        "finish_reason": "stop",
        "request_seconds": 0.1,
        "target_contract_version": TARGET_CONTRACT_VERSION,
    }


class Bp4BaseV2Tests(unittest.TestCase):
    def make_repo(self, root):
        root = Path(root)
        bp2_source = Path(
            "configs/experiments/bp2_post_training_pipeline_v1.yaml"
        )
        bp2_path = root / bp2_source
        bp2_path.parent.mkdir(parents=True, exist_ok=True)
        bp2_path.write_text(
            bp2_source.read_text(encoding="utf-8"), encoding="utf-8"
        )
        bp2 = yaml.safe_load(bp2_path.read_text(encoding="utf-8"))

        config = yaml.safe_load(
            Path(
                "configs/experiments/bp4_controlled_model_matrix_v1.yaml"
            ).read_text(encoding="utf-8")
        )
        for name in ("bp2_gate", "bp3_gate", "preference_gate", "reward_gate"):
            gate_path = root / config["pipeline"][name]["path"]
            write_json_atomic(
                gate_path,
                {"status": "PASS", "decision": f"ACCEPT_{name.upper()}"},
            )
            config["pipeline"][name]["sha256"] = file_sha256(gate_path)

        dev_rows = [asset(index, "dev") for index in range(331)]
        challenge_rows = [asset(index, "challenge") for index in range(140)]
        dev_path = root / config["evaluation"]["selection"]["path"]
        challenge_path = root / config["evaluation"]["diagnostic"]["path"]
        write_jsonl_atomic(dev_path, dev_rows)
        write_jsonl_atomic(challenge_path, challenge_rows)
        config["evaluation"]["selection"]["sha256"] = file_sha256(dev_path)
        config["evaluation"]["diagnostic"]["sha256"] = file_sha256(challenge_path)

        config_path = root / "configs/experiments/bp4_test.yaml"
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        task = base_task(config, bp2)
        task_path = root / config["outputs"]["base_task"]
        write_json_atomic(task_path, task)
        write_json_atomic(
            root / config["outputs"]["gate"],
            {
                "status": "PASS",
                "decision": "ACCEPT_BP4_MATRIX_ENGINEERING",
                "training_started": False,
            },
        )
        outputs = task["required_outputs"]
        write_jsonl_atomic(
            root / outputs["dev_predictions"],
            [prediction(row) for row in dev_rows],
        )
        write_jsonl_atomic(
            root / outputs["challenge_predictions"],
            [prediction(row) for row in challenge_rows],
        )
        output_dir = (root / outputs["base_packet"]).parent
        write_json_atomic(
            output_dir / "inference_manifest.json",
            {
                "task_sha256": file_sha256(task_path),
                "model_path": task["model_path"],
                "decoding": task["decoding"],
                "human_gold_used": False,
                "training_started": False,
            },
        )
        write_json_atomic(
            output_dir / "inference_runtime.json",
            {
                "report_version": "bp4-base-runtime-v1",
                "dev": {"rows": 331, "seconds": 10.0},
                "challenge": {"rows": 140, "seconds": 5.0},
                "peak_memory_gib": 12.0,
                "training_started": False,
            },
        )
        return config_path, task_path, config, task

    def test_dataset_contract_becomes_prompt_only_inference_rows(self):
        row = asset(1, "dev")
        prepared = prepare_inference_rows([row], "dev")
        self.assertEqual(len(prepared[0]["messages"]), 2)
        self.assertEqual(prepared[0]["messages"][-1]["role"], "user")
        self.assertEqual(prepared[0]["source"], "样品1物流很好")

    def test_completion_requires_exact_json_and_source_faithfulness(self):
        valid = parse_completion(
            "dev-1",
            '{"keywords":[["原文对象","物流",0.95]]}',
            "物流很好",
        )
        wrapped = parse_completion(
            "dev-1",
            '```json\n{"keywords":[["原文对象","物流",0.95]]}\n```',
            "物流很好",
        )
        hallucinated = parse_completion(
            "dev-1",
            '{"keywords":[["原文对象","包装",0.95]]}',
            "物流很好",
        )
        self.assertEqual(valid["status"], "success")
        self.assertEqual(wrapped["error_code"], "exact_json_parse_failed")
        self.assertEqual(hallucinated["error_code"], "target_contract_failed")

    def test_finish_reason_uses_per_sequence_eos_not_batch_padding(self):
        self.assertEqual(
            finish_reason(
                [10, 11, 2, 0, 0],
                eos_token_id=2,
                max_new_tokens=5,
            ),
            "stop",
        )
        self.assertEqual(
            finish_reason(
                [10, 11, 12, 13, 14],
                eos_token_id=2,
                max_new_tokens=5,
            ),
            "length",
        )

    def test_package_builds_complete_base_packet_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, task_path, _, task = self.make_repo(root)
            gate = package_evaluation(root, config_path, task_path)
            self.assertEqual(gate["decision"], "ACCEPT_BP4_BASE_V2")
            self.assertFalse(gate["training_started"])
            packet = json.loads(
                (root / task["required_outputs"]["base_packet"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(packet["status"], "PASS")
            self.assertEqual(packet["dev_rows"], 331)
            self.assertEqual(packet["challenge_rows"], 140)
            self.assertEqual(packet["dev_metrics"]["macro_f1"], 1.0)
            self.assertEqual(packet["challenge_usage"], "diagnostic_only")
            self.assertFalse(packet["human_gold_used"])

    def test_prediction_without_contract_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, task_path, _, task = self.make_repo(root)
            prediction_path = root / task["required_outputs"]["dev_predictions"]
            rows = json.loads(
                "[" + ",".join(prediction_path.read_text(encoding="utf-8").splitlines()) + "]"
            )
            rows[0].pop("target_contract_version")
            write_jsonl_atomic(prediction_path, rows)
            with self.assertRaisesRegex(ValueError, "frozen target contract"):
                package_evaluation(root, config_path, task_path)


if __name__ == "__main__":
    unittest.main()
