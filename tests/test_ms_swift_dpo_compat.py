import unittest

from scripts.check_ms_swift_dpo_compat import evaluate_version


class MsSwiftDpoCompatibilityTests(unittest.TestCase):
    def test_affected_releases_are_rejected(self):
        for version in ("3.11.2", "3.12.0", "3.12.1"):
            with self.subTest(version=version):
                result = evaluate_version(version)
                self.assertEqual(result["status"], "FAIL")
                self.assertEqual(
                    result["reason"],
                    "unsupported_peft_reference_adapter_loading",
                )

    def test_fixed_3_x_releases_are_accepted(self):
        for version in ("3.12.2", "3.12.3"):
            with self.subTest(version=version):
                self.assertEqual(evaluate_version(version)["status"], "PASS")

    def test_unvalidated_major_release_is_rejected(self):
        self.assertEqual(evaluate_version("4.0.0")["status"], "FAIL")

    def test_invalid_version_is_rejected(self):
        result = evaluate_version("not-a-version")
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["reason"], "invalid_ms_swift_version")


if __name__ == "__main__":
    unittest.main()
