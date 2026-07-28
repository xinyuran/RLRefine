import unittest

from scripts.package_structured_decoding_feedback import build_feedback_packet


def report(decision):
    replay = "replay"
    candidate = "candidate"
    return {
        "decision": decision,
        "sample_count": 331,
        "replay_variant": replay,
        "candidate_variant": candidate,
        "replay_fidelity_checks": {"replay_metrics_match": True},
        "candidate_checks": {
            "candidate_schema_valid_rate_at_least_99_percent": True,
            "candidate_nontruncated_outputs_are_causally_stable": True,
        },
        "metrics": {
            replay: {"macro_f1": 0.54, "schema_valid_rate": 0.98},
            candidate: {"macro_f1": 0.55, "schema_valid_rate": 1.0},
        },
        "nontruncated_exact_replay": {"rate": 1.0},
        "original_truncation_recovery": {"rescued": 8, "original_count": 8},
        "paired_candidate_minus_replay": {"bootstrap_95_ci": [0.0, 0.02]},
        "input_hashes": {"reference": "abc"},
    }


class StructuredDecodingFeedbackTests(unittest.TestCase):
    def test_adopt_authorizes_offline_replication_only(self):
        packet = build_feedback_packet(
            report("ADOPT_OFFLINE_JSON_SCHEMA_768"),
            report_sha256="report",
            manifest_sha256="manifest",
        )
        self.assertEqual(
            packet["next_work_package"], "WP2_OFFLINE_COLD_START_REPLICATION"
        )
        self.assertIn("online_concurrent_serving_not_authorized", packet["authorization"])

    def test_replay_failure_routes_to_root_cause_work(self):
        payload = report("INVALID_512_REPLAY")
        payload["replay_fidelity_checks"]["replay_metrics_match"] = False
        packet = build_feedback_packet(
            payload, report_sha256="report", manifest_sha256="manifest"
        )
        self.assertEqual(packet["next_work_package"], "WP1R_REPLAY_ROOT_CAUSE")
        self.assertEqual(packet["failed_replay_checks"], ["replay_metrics_match"])

    def test_candidate_rejection_stops_teacher_dev_token_iteration(self):
        payload = report("RETAIN_UNCONSTRAINED_B1")
        check = "candidate_nontruncated_outputs_are_causally_stable"
        payload["candidate_checks"][check] = False
        packet = build_feedback_packet(
            payload, report_sha256="report", manifest_sha256="manifest"
        )
        self.assertEqual(
            packet["next_work_package"], "WP3_CHALLENGE_SET_AND_ERROR_TAXONOMY"
        )
        self.assertEqual(packet["failed_candidate_checks"], [check])
        self.assertIn("stop_token_budget_iteration", packet["authorization"])

    def test_unknown_decision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported gate decision"):
            build_feedback_packet(
                report("MAYBE"), report_sha256="report", manifest_sha256="manifest"
            )


if __name__ == "__main__":
    unittest.main()
