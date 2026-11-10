"""Keep the dependency-free browser validator aligned with the Python contract."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from core.keyword_contract import KEYWORD_TUPLE_SIZE, MAX_KEYWORD_LENGTH, MAX_KEYWORDS
from core.target_contract import validate_keyword_payload


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node.js is needed for browser/Python parity")
class DemoContractTests(unittest.TestCase):
    def test_browser_matches_python_on_contract_boundaries(self):
        source = "包装很好，屏幕清晰。𠮷𠮷𠮷𠮷"
        item = ["原文对象", "屏幕", 0.9]
        payloads = [None, [], {}, {"keywords": []}, {"keywords": [item], "extra": 1},
                    {"keywords": [item]}, {"keywords": [item, item]},
                    {"keywords": [item] * 16}, {"keywords": [None]},
                    {"keywords": [["说明", "屏幕"]]}, {"keywords": [["说明", "屏幕", 0.9, 1]]}]
        for explanation in [None, "", " ", "\x1c", "\ufeff", "依据"]:
            for keyword in [None, "", "屏幕", "续航", "屏幕非常清晰", "𠮷𠮷𠮷𠮷"]:
                for confidence in [None, "0.9", True, False, -0.1, 0, 0.9, 1, 1.1]:
                    payloads.append({"keywords": [[explanation, keyword, confidence]]})
        script = """
import fs from 'node:fs';
import {validateKeywordPayload, MAX_KEYWORD_LENGTH, MAX_KEYWORDS, KEYWORD_TUPLE_SIZE} from './demo/contract.mjs';
const {payloads, source} = JSON.parse(fs.readFileSync(0, 'utf8'));
console.log(JSON.stringify({limits: [MAX_KEYWORD_LENGTH, MAX_KEYWORDS, KEYWORD_TUPLE_SIZE],
  results: payloads.map(payload => validateKeywordPayload(payload, source))}));
"""
        completed = subprocess.run(
            [shutil.which("node"), "--input-type=module", "-e", script],
            input=json.dumps({"payloads": payloads, "source": source}),
            cwd=ROOT, text=True, encoding="utf-8", capture_output=True, check=True,
        )
        actual = json.loads(completed.stdout)
        self.assertEqual(actual["limits"], [MAX_KEYWORD_LENGTH, MAX_KEYWORDS, KEYWORD_TUPLE_SIZE])
        self.assertEqual(len(actual["results"]), len(payloads))
        for payload, result in zip(payloads, actual["results"]):
            with self.subTest(payload=payload):
                valid, errors = validate_keyword_payload(payload, source)
                self.assertEqual(result, {"valid": valid, "errors": errors})


if __name__ == "__main__":
    unittest.main()
