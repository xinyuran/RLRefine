import json
import unittest

from core.config import Config
from core.processor import RLRefineProcessor
from examples.keyword_extraction.schema import create_keyword_task


def make_processor(responses, fallback_data=None, max_retries=0):
    processor = RLRefineProcessor.__new__(RLRefineProcessor)
    processor.config = Config(task_schema=create_keyword_task().schema, task=create_keyword_task())
    processor.config.enable_preprocess = False
    processor.config.max_retries = max_retries
    processor.task = create_keyword_task()
    processor.prompt_builder = None
    response_iter = iter(responses)
    processor._call_llm = lambda *args, **kwargs: next(response_iter, None)
    processor._fallback_extract = lambda text: fallback_data or []
    return processor


class ProcessorContractTests(unittest.TestCase):
    def test_valid_result_has_success_status_and_keeps_tuples(self):
        response = json.dumps({"keywords": [["属性", "屏幕", 0.9]]}, ensure_ascii=False)
        result = make_processor([response]).process_single("屏幕清晰", "sample-1")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"]["keywords"], [["属性", "屏幕", 0.9]])
        self.assertFalse(result["fallback"])

    def test_schema_invalid_response_is_retried(self):
        invalid = json.dumps({"keywords": [["属性", "屏幕", "0.9"]]}, ensure_ascii=False)
        valid = json.dumps({"keywords": [["属性", "屏幕", 0.9]]}, ensure_ascii=False)
        result = make_processor([invalid, valid], max_retries=1).process_single("屏幕清晰")
        self.assertEqual(result["status"], "success")

    def test_schema_failure_without_fallback_is_explicit(self):
        invalid = json.dumps({"keywords": [["属性", "屏幕", "0.9"]]}, ensure_ascii=False)
        result = make_processor([invalid]).process_single("屏幕清晰")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], "schema_validation_failed")
        self.assertTrue(result["validation_errors"])
        self.assertIsNone(result["data"])

    def test_valid_fallback_is_distinguished_from_model_success(self):
        fallback = [["jieba-fallback", "屏幕", 0.6]]
        result = make_processor([None], fallback_data=fallback).process_single("屏幕清晰")
        self.assertEqual(result["status"], "fallback")
        self.assertEqual(result["primary_error_code"], "extraction_failed")
        self.assertEqual(result["data"]["keywords"], fallback)

    def test_postprocess_schema_failure_does_not_return_invalid_data(self):
        processor = make_processor([], fallback_data=[])
        processor.config.post_process_return_full_info = False
        result = processor._finalize_result(
            {"keywords": [["属性", "屏幕", 0.9]]},
            "屏幕清晰",
            "sample-1",
            fallback=False,
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], "postprocess_schema_validation_failed")
        self.assertIsNone(result["data"])


if __name__ == "__main__":
    unittest.main()
