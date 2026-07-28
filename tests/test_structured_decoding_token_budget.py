import unittest
from copy import deepcopy

from evaluation.structured_decoding_inference import (
    CONSTRAINED_VARIANT,
    CONTROL_VARIANT,
    build_request,
)
from evaluation.structured_decoding_offline_token_budget_inference import (
    CANDIDATE_VARIANT as OFFLINE_CANDIDATE_VARIANT,
    REPLAY_VARIANT as OFFLINE_REPLAY_VARIANT,
    build_chat_prompts,
)
from evaluation.structured_decoding_token_budget_diagnostics import analyze
from evaluation.structured_decoding_token_budget_inference import (
    CANDIDATE_VARIANT,
    REPLAY_VARIANT,
)


def prediction(
    sample_id,
    variant,
    keyword=None,
    *,
    output_tokens=20,
    finish_reason="stop",
    raw_response=None,
):
    success = keyword is not None
    raw = raw_response if raw_response is not None else (
        f'{{"keywords":[["依据","{keyword}",0.9]]}}' if success else '{"keywords":['
    )
    return {
        "sample_id": sample_id,
        "status": "success" if success else "error",
        "data": {"keywords": [["依据", keyword, 0.9]]} if success else None,
        "error_code": None if success else "json_parse_failed",
        "prompt_variant": variant,
        "raw_response": raw,
        "input_tokens": 100,
        "output_tokens": output_tokens,
        "finish_reason": finish_reason,
        "request_seconds": 0.1,
    }


class StructuredDecodingTokenBudgetTests(unittest.TestCase):
    def setUp(self):
        self.reference = [
            {"sample_id": "a", "normalized_text": "物流很快", "gold": {"keywords": ["物流"]}},
            {"sample_id": "b", "normalized_text": "包装很好", "gold": {"keywords": ["包装"]}},
        ]
        self.control = [
            prediction("a", CONTROL_VARIANT, "物流", raw_response="stable-a"),
            prediction("b", CONTROL_VARIANT, "包装", raw_response="stable-b"),
        ]
        self.prior = [
            prediction("a", CONSTRAINED_VARIANT, "物流", raw_response="stable-a"),
            prediction(
                "b", CONSTRAINED_VARIANT, None,
                output_tokens=512, finish_reason="length", raw_response="truncated-b",
            ),
        ]
        self.replay = [
            prediction("a", REPLAY_VARIANT, "物流", raw_response="stable-a"),
            prediction(
                "b", REPLAY_VARIANT, None,
                output_tokens=512, finish_reason="length", raw_response="truncated-b",
            ),
        ]
        self.candidate = [
            prediction("a", CANDIDATE_VARIANT, "物流", raw_response="stable-a"),
            prediction("b", CANDIDATE_VARIANT, "包装", raw_response="completed-b"),
        ]

    def test_only_max_tokens_differs_between_replay_and_candidate(self):
        replay_request = build_request("物流很快", "model", 512, 42, constrained=True)
        candidate_request = build_request("物流很快", "model", 768, 42, constrained=True)
        self.assertEqual(replay_request.pop("max_tokens"), 512)
        self.assertEqual(candidate_request.pop("max_tokens"), 768)
        self.assertEqual(replay_request, candidate_request)

    def test_offline_chat_prompt_uses_frozen_b1_messages(self):
        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                self.messages = messages
                self.options = (tokenize, add_generation_prompt)
                return "rendered"

        tokenizer = FakeTokenizer()
        prompts = build_chat_prompts(
            tokenizer, [{"sample_id": "a", "normalized_text": "物流很快"}]
        )
        self.assertEqual(prompts, ["rendered"])
        self.assertEqual([item["role"] for item in tokenizer.messages], ["system", "user"])
        self.assertIn("物流很快", tokenizer.messages[1]["content"])
        self.assertEqual(tokenizer.options, (False, True))

    def test_768_candidate_that_only_rescues_truncation_is_adopted(self):
        report, rows = analyze(
            self.reference,
            self.control,
            self.prior,
            self.replay,
            self.candidate,
            bootstrap_iterations=100,
            seed=42,
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(report["replay_fidelity_pass"])
        self.assertTrue(report["candidate_pass"])
        self.assertEqual(report["original_truncation_recovery"]["rescued"], 1)
        self.assertEqual(report["decision"], "ADOPT_JSON_SCHEMA_768")

    def test_replay_drift_invalidates_experiment(self):
        self.replay[0] = prediction(
            "a", REPLAY_VARIANT, "很快", raw_response="drifted-a"
        )
        report, _ = analyze(
            self.reference,
            self.control,
            self.prior,
            self.replay,
            self.candidate,
            bootstrap_iterations=100,
            seed=42,
        )
        self.assertFalse(report["replay_fidelity_pass"])
        self.assertEqual(report["decision"], "INVALID_512_REPLAY")

    def test_dynamic_offline_variant_names_use_same_gate(self):
        replay = deepcopy(self.replay)
        candidate = deepcopy(self.candidate)
        for row in replay:
            row["prompt_variant"] = OFFLINE_REPLAY_VARIANT
        for row in candidate:
            row["prompt_variant"] = OFFLINE_CANDIDATE_VARIANT
        report, _ = analyze(
            self.reference,
            self.control,
            self.prior,
            replay,
            candidate,
            bootstrap_iterations=100,
            seed=42,
            replay_variant=OFFLINE_REPLAY_VARIANT,
            candidate_variant=OFFLINE_CANDIDATE_VARIANT,
        )
        self.assertTrue(report["candidate_pass"])
        self.assertEqual(report["replay_variant"], OFFLINE_REPLAY_VARIANT)
        self.assertEqual(report["candidate_variant"], OFFLINE_CANDIDATE_VARIANT)


if __name__ == "__main__":
    unittest.main()
