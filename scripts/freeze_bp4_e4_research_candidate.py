"""Freeze E4 after teacher-dev selection, without rewriting its historical gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.keyword_evaluator import file_sha256, write_json_atomic


def same_checkpoint(candidate: object, latest: object) -> bool:
    """Compare lineage paths after resolving relative-path spelling differences."""
    return (
        isinstance(candidate, str)
        and isinstance(latest, str)
        and Path(candidate).resolve() == Path(latest).resolve()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--paired-packet", type=Path, required=True)
    parser.add_argument("--training-artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        gate = json.loads(args.gate.read_text(encoding="utf-8"))
        packet = json.loads(args.paired_packet.read_text(encoding="utf-8"))
        artifacts = json.loads(args.training_artifacts.read_text(encoding="utf-8"))
        checkpoint = packet.get("candidate_checkpoint")
        latest = (artifacts.get("latest_checkpoint") or {}).get("checkpoint")
        candidate = packet.get("candidate") or {}
        baseline = packet.get("baseline") or {}
        checks = {
            "stage_is_e4": gate.get("stage") == "grpo" and packet.get("candidate_model_id") == "E4_GRPO_FROM_DPO",
            "same_final_checkpoint": same_checkpoint(checkpoint, latest),
            "teacher_dev_only_selection": packet.get("selection_split") == "teacher_dev" and packet.get("human_gold_used") is False,
            "challenge_diagnostic_only": packet.get("challenge_usage") == "diagnostic_only",
            "e4_improves_over_dpo_macro_f1": candidate.get("macro_f1", -1) > baseline.get("macro_f1", -1),
            "e4_improves_over_dpo_schema": candidate.get("schema_valid_rate", -1) > baseline.get("schema_valid_rate", -1),
        }
        if not all(checks.values()):
            raise ValueError(f"E4 candidate freeze checks failed: {[name for name, value in checks.items() if not value]}")
        payload: dict[str, Any] = {
            "report_version": "bp5-e4-research-candidate-freeze-v1",
            "status": "PASS",
            "decision": "FREEZE_E4_AS_OPTIMAL_RESEARCH_CANDIDATE",
            "candidate_model_id": "E4_GRPO_FROM_DPO",
            "candidate_checkpoint": checkpoint,
            "selection_basis": "frozen_teacher_dev_only",
            "human_gold_used_for_selection": False,
            "challenge_used_for_selection": False,
            "historical_e4_gate": {"path": str(args.gate), "sha256": file_sha256(args.gate), "research_progression_decision": gate.get("research_progression_decision"), "deployment_decision": gate.get("deployment_decision")},
            "paired_packet": {"path": str(args.paired_packet), "sha256": file_sha256(args.paired_packet)},
            "training_artifacts": {"path": str(args.training_artifacts), "sha256": file_sha256(args.training_artifacts)},
            "checks": checks,
            "next_execution": "BP5_DPO_E4_HUMAN_GOLD_ONE_SHOT",
            "deployment_authorized": False,
        }
        write_json_atomic(args.output, payload)
        print(json.dumps({"event": "bp5_e4_candidate_freeze_complete", **payload}, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"event": "bp5_e4_candidate_freeze_complete", "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
