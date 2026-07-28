import csv
import tempfile
import unittest
from pathlib import Path

from evaluation.keyword_evaluator import read_jsonl
from scripts.build_second_schema_assets import run as build_assets
from scripts.freeze_second_schema_dataset import run as freeze_assets


class SecondSchemaAssetTests(unittest.TestCase):
    def test_draft_counts_contract_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "draft"
            manifest = build_assets(output)
            self.assertEqual(manifest["validation"]["split_counts"], {"dev": 120, "test": 100, "challenge": 20})
            self.assertEqual(manifest["validation"]["group_overlap_count"], 0)
            self.assertFalse(manifest["human_verified"])
            rows = read_jsonl(output / "dev.jsonl")
            self.assertTrue(all(row["target"]["evidence"] in row["source"] for row in rows))
            self.assertTrue(all(row["annotation_status"] == "draft" for row in rows))

    def test_freeze_rejects_unapproved_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_assets(root / "draft")
            with self.assertRaisesRegex(ValueError, "is not approved"):
                freeze_assets(root / "draft" / "review_packet.csv", root / "frozen", "reviewer", "2026-07-28")

    def test_fully_approved_packet_can_be_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_assets(root / "draft")
            packet = root / "draft" / "review_packet.csv"
            with packet.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
                fields = list(rows[0])
            for row in rows:
                row["approved"] = "true"
            with packet.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            manifest = freeze_assets(packet, root / "frozen", "reviewer", "2026-07-28")
            self.assertTrue(manifest["frozen"])
            self.assertTrue(manifest["human_verified"])
            frozen_rows = read_jsonl(root / "frozen" / "test.jsonl")
            self.assertEqual(len(frozen_rows), 100)
            self.assertTrue(all(row["human_verified"] for row in frozen_rows))

    def test_freeze_rejects_immutable_source_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_assets(root / "draft")
            packet = root / "draft" / "review_packet.csv"
            with packet.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
                fields = list(rows[0])
            for row in rows:
                row["approved"] = "true"
            rows[0]["source"] += "擅自修改"
            with packet.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "changed immutable field: source"):
                freeze_assets(packet, root / "frozen", "reviewer", "2026-07-28")


if __name__ == "__main__":
    unittest.main()
