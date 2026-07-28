import unittest

from scripts.split_canonical_data import split_records, verify_disjoint


def records(count):
    return [{"sample_id": f"sample-{index:03d}", "messages": []} for index in range(count)]


class CanonicalDataSplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_input_order_independent(self):
        source = records(20)

        first = split_records(source, seed="seed")
        second = split_records(list(reversed(source)), seed="seed")

        for split_name in first:
            self.assertEqual(
                [item["sample_id"] for item in first[split_name]],
                [item["sample_id"] for item in second[split_name]],
            )

    def test_default_ratios_have_expected_group_counts(self):
        result = split_records(records(20), seed="seed")

        self.assertEqual(len(result["train"]), 15)
        self.assertEqual(len(result["dev"]), 2)
        self.assertEqual(len(result["test_candidate"]), 3)

    def test_same_group_never_crosses_splits(self):
        source = records(20)
        source[0]["group_id"] = "shared-group"
        source[1]["group_id"] = "shared-group"

        result = split_records(source, seed="seed")
        locations = {
            item["sample_id"]: split_name
            for split_name, items in result.items()
            for item in items
        }

        self.assertEqual(locations["sample-000"], locations["sample-001"])
        self.assertTrue(all(value == 0 for value in verify_disjoint(result).values()))

    def test_duplicate_sample_id_is_rejected(self):
        source = records(2)
        source[1]["sample_id"] = source[0]["sample_id"]

        with self.assertRaisesRegex(ValueError, "Duplicate sample_id"):
            split_records(source)

    def test_missing_sample_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sample_id"):
            split_records([{"messages": []}])

    def test_invalid_ratios_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            split_records(records(3), train_ratio=0.8, dev_ratio=0.1, test_ratio=0.2)


if __name__ == "__main__":
    unittest.main()
