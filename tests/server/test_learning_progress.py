"""Learning milestones, visit accounting, and honest estimates for the console."""

import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

from crossphase_miner.core.models import PeriodModel, PhaseTransition, SignalColor
from crossphase_miner.core.tod import classify_tod
from server import config
from server.service import SignalService


class LearningProgressTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = str(Path(self.directory.name) / "state.sqlite3")
        self.service = self.open_service()
        self.iid = self.service.order[0]
        self.now = self.service.clock.now()
        self.period = classify_tod(self.now, self.service.period_configs)

    def open_service(self):
        return SignalService(
            num_intersections=12,
            zone_size=8,
            num_hubs=2,
            speed=0,
            sim_start=config.default_sim_start() + 3600,
            db_path=self.path,
        )

    def tearDown(self):
        self.service.shutdown()
        self.directory.cleanup()

    def summary(self):
        return self.service.detail(self.iid)["learning"]

    def visit(self, robot="robot_001", mode="scout", action="CROSS", offset=0, **kwargs):
        return self.service.submit_arrival(
            robot,
            self.iid,
            self.now + offset,
            self.now + offset + 60,
            mode,
            kwargs.pop("transitions", []),
            60,
            action=action,
            **kwargs,
        )

    def test_empty_model_has_planning_estimate_not_confidence(self):
        result = self.summary()
        self.assertIsNone(result["confidence"])
        self.assertEqual(result["history"], {})
        progress = result["progress"]
        self.assertEqual(progress["remaining_samples"], 6)
        self.assertEqual(progress["sample_progress"], 0)
        self.assertEqual(progress["estimated_scout_visits"], [2, 2])
        self.assertEqual(progress["estimate_basis"], "planning")
        self.assertIsNone(progress["mean_transitions_per_scout"])

    def test_unique_robots_repeated_visits_modes_and_timeouts_survive_restart(self):
        self.visit(record_id="first")
        self.assertTrue(self.visit(record_id="first")["duplicate"])
        self.visit(mode="normal", offset=100)
        self.assertTrue(self.visit(mode="normal", offset=100)["duplicate"])
        self.visit(robot="robot_002", action="TIMEOUT", offset=200)
        expected = dict(robots=2, visits=3, scout=2, normal=1, cross=2, timeout=1)
        self.assertEqual(self.summary()["visits"], expected)
        robot = self.service.robots["robot_002"]
        self.assertEqual((robot.action, robot.crossings), ("TIMEOUT", 0))
        self.service.shutdown()
        self.service = self.open_service()
        self.assertEqual(self.summary()["visits"], expected)
        self.assertEqual(len(self.service.detail(self.iid)["recent_visits"]), 3)
        progress = self.summary()["progress"]
        self.assertEqual(progress["estimate_basis"], "no_transitions")
        self.assertEqual(progress["recent_scout_visits"], 2)

    def test_historical_estimate_includes_zero_yield_and_only_relevant_scouts(self):
        other = "night" if self.period == "day" else "day"
        self.service.visit_yields[self.iid] = deque(
            [
                dict(
                    mode="scout",
                    period=self.period,
                    independent_transitions_by_period={self.period: 4},
                ),
                dict(mode="scout", period=self.period, independent_transitions_by_period={}),
                dict(
                    mode="normal",
                    period=self.period,
                    independent_transitions_by_period={self.period: 10},
                ),
                dict(mode="scout", period=other, independent_transitions_by_period={other: 4}),
            ]
        )
        self.service.models[(self.iid, self.period)] = PeriodModel(sample_count=2, confidence=0.8)
        progress = self.summary()["progress"]
        self.assertEqual(progress["remaining_samples"], 4)
        self.assertAlmostEqual(progress["sample_progress"], 1 / 3)
        self.assertEqual(progress["estimated_scout_visits"], [2, 2])
        self.assertEqual(progress["mean_transitions_per_scout"], 2)
        self.assertEqual(progress["recent_scout_visits"], 2)
        self.assertEqual(progress["estimate_basis"], "history")
        night = self.service.learning_progress(self.iid, other, None)
        self.assertEqual(night["remaining_samples"], 6)
        self.assertEqual(night["estimated_scout_visits"], [2, 2])

    def test_sample_completion_does_not_claim_reliability_or_fixed_visits(self):
        key = self.iid, self.period
        self.service.models[key] = PeriodModel(sample_count=6, confidence=0.2)
        summary = self.summary()
        self.assertEqual(summary["status"], "learning")
        self.assertEqual(summary["progress"]["sample_progress"], 1)
        self.assertEqual(summary["progress"]["estimate_basis"], "confidence")
        self.assertIsNone(summary["progress"]["estimated_scout_visits"])
        self.service.models[key].confidence = 0.3
        summary = self.summary()
        self.assertEqual(summary["status"], "reliable")
        self.assertEqual(summary["progress"]["estimated_scout_visits"], [0, 0])

    def test_actual_learning_milestones_and_estimate_survive_restart(self):
        transitions = [
            PhaseTransition(
                intersection_id=self.iid,
                timestamp=self.now - (12 - i) * 60,
                from_color=SignalColor.RED if i % 2 == 0 else SignalColor.GREEN,
                to_color=SignalColor.GREEN if i % 2 == 0 else SignalColor.RED,
                robot_id="robot_001",
                episode_start=self.now - (12 - i) * 60 - 90,
            )
            for i in range(12)
        ]
        self.service._learn(self.iid, transitions[:3])
        first = self.summary()["history"]["first_model"]
        self.assertEqual(first["samples"], 3)
        self.assertNotIn("first_reliable", self.summary()["history"])
        self.service._learn(self.iid, transitions[3:5])
        self.assertEqual(self.summary()["status"], "learning")
        self.assertNotIn("first_reliable", self.summary()["history"])
        self.service._learn(self.iid, transitions[5:6])
        summary = self.summary()
        self.assertEqual(summary["status"], "reliable")
        self.assertEqual(summary["history"]["first_reliable"]["samples"], 6)
        self.assertEqual(summary["history"]["first_model"], first)
        reliable = summary["history"]["first_reliable"]
        self.service._learn(self.iid, transitions[6:])
        self.assertEqual(self.summary()["history"]["last_update"]["samples"], 12)
        self.assertEqual(self.summary()["history"]["first_reliable"], reliable)
        before = self.summary()
        self.service.shutdown()
        self.service = self.open_service()
        self.assertEqual(self.summary(), before)

    def test_existing_models_do_not_acquire_fabricated_creation_time(self):
        key = self.iid, self.period
        learner = self.service.learners.get(key)
        learner.sample_count = 12
        self.service.models[key] = learner.get_model()
        self.service._learn(self.iid, [])
        history = self.summary()["history"]
        self.assertNotIn("first_model", history)
        self.assertNotIn("first_reliable", history)
        self.assertIn("last_update", history)

    def test_learn_worker_survives_exception_and_keeps_processing(self):
        """A failure inside one learning update must not kill the worker
        thread; the queue should keep draining afterwards."""
        original = self.service._apply_shared_green
        calls = {"n": 0}

        def flaky(intersection_id, *args):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return original(intersection_id, *args)

        transitions = [dict(timestamp=self.now, from_color="RED", to_color="GREEN")]
        with patch.object(self.service, "_apply_shared_green", side_effect=flaky):
            self.visit(record_id="boom", transitions=transitions)
            self.service.wait_for_learning()

        self.assertEqual(self.service.stats.learn_errors, 1)
        self.assertTrue(self.service._worker.is_alive())
        # pending_learning must still be decremented on failure, not stuck.
        self.assertEqual(self.summary()["progress"]["pending_updates"], 0)

        # The worker must still be alive and processing after the failure.
        self.visit(record_id="after", offset=60, transitions=transitions)
        self.service.wait_for_learning()
        self.assertEqual(self.service.stats.learn_errors, 1)
        self.assertEqual(self.summary()["progress"]["pending_updates"], 0)
        self.assertGreaterEqual(self.summary()["progress"]["sample_progress"], 0)

    def test_reports_remain_pending_when_worker_is_unavailable(self):
        self.service._stop.set()
        self.service._learn_ready.set()
        self.service._worker.join()
        transitions = [dict(timestamp=self.now, from_color="RED", to_color="GREEN")]
        result = self.visit(record_id="pending", transitions=transitions)
        self.assertTrue(result["stored"])
        self.assertEqual(result["queued"], 1)
        self.assertEqual(self.summary()["progress"]["pending_updates"], 1)
        result = self.visit(record_id="also-pending", transitions=transitions)
        self.assertEqual(result["queued"], 1)
        self.assertNotIn("dropped", result)
        self.assertEqual(self.service.store.pending_jobs()[self.iid], 2)


if __name__ == "__main__":
    unittest.main()
