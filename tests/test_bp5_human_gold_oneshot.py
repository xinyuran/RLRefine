import tempfile
import unittest
from pathlib import Path

from evaluation.bp5_human_gold_oneshot import same_adapter_path


class Bp5HumanGoldOneShotTests(unittest.TestCase):
    def test_adapter_comparison_normalizes_relative_path_spelling(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint-295"
            checkpoint.mkdir()
            alternate = checkpoint.parent / "nested" / ".." / checkpoint.name
            self.assertTrue(same_adapter_path(checkpoint, alternate))

    def test_adapter_comparison_rejects_different_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "checkpoint-74"; second = root / "checkpoint-295"
            first.mkdir(); second.mkdir()
            self.assertFalse(same_adapter_path(first, second))


if __name__ == "__main__":
    unittest.main()
