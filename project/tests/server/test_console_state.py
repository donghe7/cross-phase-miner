"""Regression coverage for reported locations and raw observation confidence."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crossphase_miner.core.models import PeriodModel
from crossphase_miner.core.tod import classify_tod
from server import config
from server.service import SignalService


class ConsoleStateTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.service = SignalService(
            num_intersections=12,
            zone_size=8,
            num_hubs=2,
            speed=0,
            db_path=str(Path(self.directory.name) / "test.sqlite3"),
        )
        self.now = self.service.clock.now()
        self.origin, self.target = self.service.order[:2]

    def tearDown(self):
        self.service.shutdown()
        self.directory.cleanup()

    def robot(self):
        return self.service.snapshot()["robots"][0]

    def test_travel_clears_old_location_and_observation(self):
        self.service.ingest_batch("robot_001", self.origin, [(self.now, "RED", 0.95)])
        self.service.report_travel("robot_001", self.target, self.now, self.now + 90)
        robot = self.robot()
        self.assertEqual(robot["origin_id"], self.origin)
        self.assertEqual(robot["destination_id"], self.target)
        self.assertEqual(robot["action"], "TRAVEL")
        self.assertIsNone(robot["intersection_id"])
        self.assertIsNone(robot["observation"])
        self.assertNotIn("robot_001", self.service.detail(self.origin)["robots_present"])
        self.service.ingest_batch("robot_001", self.target, [(self.now + 90, "GREEN", 0.96)])
        robot = self.robot()
        self.assertEqual(robot["intersection_id"], self.target)
        self.assertIsNone(robot["destination_id"])
        self.assertIsNone(robot["travel_arrives_at"])
        self.assertEqual(robot["observation"]["confidence"], 0.96)

    def test_rejected_frame_visible_without_becoming_a_vote(self):
        with patch.object(config, "OBS_CONFIDENCE_THRESHOLD", 0.83):
            self.service.ingest_batch(
                "robot_001", self.origin, [(self.now - 1, "RED", 0.95), (self.now, "GREEN", 0.72)]
            )
            frame = self.robot()["observation"]
            self.assertEqual(frame["color"], "GREEN")
            self.assertEqual(frame["confidence"], 0.72)
            self.assertFalse(frame["accepted"])
            detail = self.service.detail(self.origin)
            self.assertEqual(detail["color"], "RED")
            self.assertEqual(detail["live"]["voters"], 1)
            self.assertEqual(detail["live"]["agreement"], 1)
            self.assertEqual(detail["live"]["rejected"], 1)
            self.service.ingest_batch("robot_001", self.origin, [(self.now, "GREEN", 0.83)])
            self.assertTrue(self.robot()["observation"]["accepted"])

    def test_startup_trip_has_unknown_origin(self):
        self.service.report_travel("robot_001", self.target, self.now, self.now + 90)
        self.assertIsNone(self.robot()["origin_id"])
        self.assertIsNone(self.robot()["intersection_id"])
        self.assertEqual(self.robot()["destination_id"], self.target)

    def test_latest_frame_is_chosen_by_timestamp(self):
        self.service.ingest_batch(
            "robot_001", self.origin, [(self.now, "RED", 0.92), (self.now - 1, "GREEN", 0.99)]
        )
        self.assertEqual(self.robot()["observation"]["color"], "RED")

    def test_unknown_destination_does_not_change_location(self):
        self.service.ingest_batch("robot_001", self.origin, [(self.now, "RED", 0.95)])
        with self.assertRaises(KeyError):
            self.service.report_travel("robot_001", "missing", self.now, self.now + 90)
        self.assertEqual(self.robot()["intersection_id"], self.origin)

    def test_reliable_model_does_not_hide_live_disagreement(self):
        period = classify_tod(self.now, self.service.period_configs)
        self.service.models[(self.origin, period)] = PeriodModel(confidence=0.95, sample_count=20)
        self.service.ingest_batch("robot_001", self.origin, [(self.now, "RED", 0.95)])
        self.service.ingest_batch("robot_002", self.origin, [(self.now, "GREEN", 0.95)])
        detail = self.service.detail(self.origin)
        self.assertEqual(detail["source"], "disputed")
        self.assertIsNone(detail["color"])
        self.assertIsNone(detail["seconds_to_green"])
        self.assertEqual(self.service.snapshot()["tiles"][0][0], 5)
        self.assertIn(detail["model_prediction"]["color"], ("RED", "GREEN"))

    def test_observation_and_prediction_keep_their_own_colors_and_countdown(self):
        period = classify_tod(self.now, self.service.period_configs)
        self.service.models[(self.origin, period)] = PeriodModel(
            confidence=0.95,
            sample_count=20,
            T_cycle=120,
            T_red=90,
            T_green=30,
            phi_offset=self.now - 100,
        )
        self.service.ingest_batch("robot_001", self.origin, [(self.now, "RED", 0.95)])
        detail = self.service.detail(self.origin)
        self.assertEqual(detail["live"]["color"], "RED")
        self.assertEqual(detail["source"], "observed")
        self.assertEqual(
            detail["model_prediction"],
            dict(color="GREEN", seconds_to_green=None, green_remaining=20.0),
        )
        self.service.clock.sim_start += config.LIVE_STALE_SECONDS + 1
        detail = self.service.detail(self.origin)
        self.assertIsNone(detail["live"]["color"])
        self.assertIn(detail["model_prediction"]["color"], ("RED", "GREEN"))

    def test_unreliable_model_does_not_claim_predicted_color(self):
        period = classify_tod(self.now, self.service.period_configs)
        self.service.models[(self.origin, period)] = PeriodModel(confidence=0.9, sample_count=3)
        self.service.ingest_batch("robot_001", self.origin, [(self.now, "GREEN", 0.95)])
        detail = self.service.detail(self.origin)
        self.assertEqual(detail["live"]["color"], "GREEN")
        self.assertEqual(
            detail["model_prediction"],
            dict(color=None, seconds_to_green=None, green_remaining=None),
        )


if __name__ == "__main__":
    unittest.main()
