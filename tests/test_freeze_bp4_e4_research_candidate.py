import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.freeze_bp4_e4_research_candidate import same_checkpoint


class FreezeE4ResearchCandidateTests(unittest.TestCase):
    def test_checkpoint_comparison_normalizes_path_spelling(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint-295"
            checkpoint.mkdir()
            self.assertTrue(same_checkpoint(str(checkpoint), str(checkpoint.parent / "nested" / ".." / checkpoint.name)))

    def test_freezes_e4_without_rewriting_historical_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate = root / "gate.json"; packet = root / "packet.json"; artifacts = root / "artifacts.json"; output = root / "freeze.json"
            gate.write_text(json.dumps({"stage": "grpo", "research_progression_decision": "REJECT_E4_GRPO_RESEARCH_PROGRESSION", "deployment_decision": "REJECT_E4_GRPO_DEPLOYMENT"}), encoding="utf-8")
            packet.write_text(json.dumps({"candidate_model_id": "E4_GRPO_FROM_DPO", "candidate_checkpoint": "checkpoint-295", "selection_split": "teacher_dev", "human_gold_used": False, "challenge_usage": "diagnostic_only", "baseline": {"macro_f1": .5, "schema_valid_rate": .74}, "candidate": {"macro_f1": .53, "schema_valid_rate": .79}}), encoding="utf-8")
            artifacts.write_text(json.dumps({"latest_checkpoint": {"checkpoint": "checkpoint-295"}}), encoding="utf-8")
            result = subprocess.run([sys.executable, "-m", "scripts.freeze_bp4_e4_research_candidate", "--gate", str(gate), "--paired-packet", str(packet), "--training-artifacts", str(artifacts), "--output", str(output)], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout)
            frozen = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(frozen["decision"], "FREEZE_E4_AS_OPTIMAL_RESEARCH_CANDIDATE")
            self.assertEqual(frozen["historical_e4_gate"]["deployment_decision"], "REJECT_E4_GRPO_DEPLOYMENT")


if __name__ == "__main__":
    unittest.main()
