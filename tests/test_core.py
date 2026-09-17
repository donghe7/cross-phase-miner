"""
Integration tests for CrossPhaseMiner.
Tests the offline simulator and causal period learning pipeline.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

import numpy as np

from crossphase_miner.core.learners import BayesianPeriodLearner, UKFPeriodLearner
from crossphase_miner.core.models import (
    PeriodModel,
    PhaseTransition,
    SignalColor,
    SignalObservation,
)
from crossphase_miner.core.tod import SIMPLE_TOD_PERIODS, TODManager
from crossphase_miner.core.transitions import extract_transitions
from crossphase_miner.pipeline.online import OnlinePipeline
from crossphase_miner.simulation.robot import RobotArrivalSimulator
from crossphase_miner.simulation.world import SignalWorld

# Test cycle config (day/night identical for simplicity).
TEST_CYCLE_CONFIG = {
    "day": {"T_cycle": 120, "T_red": 80, "T_green": 40, "phase_offset": 0},
    "night": {"T_cycle": 120, "T_red": 80, "T_green": 40, "phase_offset": 0},
}

# KST 2024-07-16 00:00:00 = UTC 2024-07-15 15:00:00
BASE = datetime(2024, 7, 15, 15, 0, 0, tzinfo=timezone.utc).timestamp()


def _make_world(iid: str = "test_001") -> SignalWorld:
    world = SignalWorld(period_configs=SIMPLE_TOD_PERIODS)
    world.add_intersection(iid, TEST_CYCLE_CONFIG)
    return world


class TestModels(unittest.TestCase):
    def test_period_model_prediction(self):
        # phi_offset=0 -> cycle origin (red start) at t=0; red [0,80), green [80,120)
        model = PeriodModel(
            T_cycle=120, T_red=80, T_green=40, phi_offset=0, confidence=0.9, sample_count=20
        )
        next_green, conf = model.predict_next_green(0)
        self.assertAlmostEqual(next_green, 80, delta=1)
        self.assertGreater(conf, 0.5)
        self.assertEqual(model.predict_color(10), SignalColor.RED)
        self.assertEqual(model.predict_color(90), SignalColor.GREEN)

    def test_period_model_low_confidence(self):
        model = PeriodModel(confidence=0.1, sample_count=2)
        next_green, conf = model.predict_next_green(0)
        self.assertEqual(next_green, float("inf"))


class TestSimulation(unittest.TestCase):
    """Test offline signal simulation."""

    def test_world_color(self):
        world = _make_world()
        # phase_offset=0: red during [0, 80) of each 120s cycle from KST midnight
        self.assertEqual(world.color_at("test_001", BASE + 10), SignalColor.RED)
        self.assertEqual(world.color_at("test_001", BASE + 90), SignalColor.GREEN)

    def test_world_next_green(self):
        world = _make_world()
        next_green = world.next_color_time("test_001", BASE + 10, SignalColor.GREEN)
        self.assertAlmostEqual(next_green, BASE + 80, delta=1e-6)

    def test_robot_arrival(self):
        world = _make_world()
        sim = RobotArrivalSimulator(world, random_seed=42)

        obs = sim.simulate_arrival("test_001", BASE + 5, "robot_001", misclass_prob=0.0)

        self.assertGreater(len(obs), 0)
        # Default rate is 10Hz: observations are 0.1s apart.
        for i in range(1, len(obs)):
            self.assertAlmostEqual(obs[i].timestamp - obs[i - 1].timestamp, 0.1, delta=0.001)
        self.assertEqual(obs[0].robot_id, "robot_001")
        self.assertEqual(obs[0].intersection_id, "test_001")
        self.assertIn(obs[0].color, [SignalColor.RED, SignalColor.GREEN])

    def test_robot_crosses_after_strict_confirmation(self):
        world = _make_world()
        sim = RobotArrivalSimulator(world, random_seed=42)
        # Arrive at green start: with the model-free baseline the robot needs
        # default_confirm_threshold (5) seconds of GREEN = 50 obs at 10Hz.
        obs = sim.simulate_arrival("test_001", BASE + 80, "robot_001", misclass_prob=0.0)
        self.assertEqual(len(obs), 50)

    def test_misclassification(self):
        world = _make_world()
        sim = RobotArrivalSimulator(world, random_seed=42)
        obs_clean = sim.simulate_arrival("test_001", BASE + 5, "r1", misclass_prob=0.0)

        sim2 = RobotArrivalSimulator(_make_world(), random_seed=42)
        obs_noisy = sim2.simulate_arrival("test_001", BASE + 5, "r1", misclass_prob=1.0)

        for o_clean, o_noisy in zip(obs_clean, obs_noisy):
            if o_clean.color == SignalColor.RED:
                self.assertEqual(o_noisy.color, SignalColor.GREEN)
            else:
                self.assertEqual(o_noisy.color, SignalColor.RED)

    def test_transition_extraction(self):
        obs = []
        for i in range(100):
            color = SignalColor.RED if i < 40 else SignalColor.GREEN
            obs.append(SignalObservation("test", i, color, 0.95, "r1"))

        trans = extract_transitions(obs, min_duration=2.0)
        self.assertEqual(len(trans), 1)
        self.assertEqual(trans[0].from_color, SignalColor.RED)
        self.assertEqual(trans[0].to_color, SignalColor.GREEN)
        # Backdated to the first second the new color was seen (index 40).
        self.assertEqual(trans[0].timestamp, 40)
        self.assertEqual(trans[0].episode_start, 0)

    def test_transition_debouncing(self):
        obs = []
        for i in range(20):
            color = SignalColor.RED
            if i == 10:  # Single green sample (noise)
                color = SignalColor.GREEN
            obs.append(SignalObservation("test", i, color, 0.95, "r1"))

        trans = extract_transitions(obs, min_duration=2.0)
        self.assertEqual(len(trans), 0)


class TestPeriodLearner(unittest.TestCase):
    def _make_rg_transitions(self, T_cycle=120, T_red=80, n=20):
        """RED->GREEN transitions every T_cycle seconds, with episode starts
        spread over the red phase (uniform-arrival assumption)."""
        transitions = []
        t_rg = 90.0  # red starts at 10, green at 90 for T_red=80
        for i in range(n):
            t = t_rg + i * T_cycle
            wait = 5.0 + (T_red - 5.0) * (i % 10) / 9.0
            transitions.append(
                PhaseTransition(
                    "test_001",
                    t,
                    SignalColor.RED,
                    SignalColor.GREEN,
                    episode_start=t - wait,
                )
            )
        return transitions

    def test_bayesian_learning(self):
        trans = self._make_rg_transitions(T_cycle=120, T_red=80, n=20)
        learner = BayesianPeriodLearner()
        for t in trans:
            model = learner.update(t)
        self.assertAlmostEqual(model.T_cycle, 120, delta=15)
        self.assertAlmostEqual(model.T_red, 80, delta=25)
        self.assertGreater(model.confidence, 0.3)
        # phi_offset is a cycle origin (red start); green starts T_red later.
        self.assertEqual(model.predict_color(90.5), SignalColor.GREEN)

    def test_ukf_learning(self):
        trans = self._make_rg_transitions(T_cycle=120, T_red=80, n=20)
        ukf = UKFPeriodLearner()
        for t in trans:
            model = ukf.update(t)
        self.assertAlmostEqual(model.T_cycle, 120, delta=15)


class TestTODManager(unittest.TestCase):
    def test_classify_tod(self):
        mgr = TODManager()  # default 4-period Korean TOD
        self.assertEqual(mgr.classify_tod(BASE + 7.5 * 3600), "morning_rush")
        self.assertEqual(mgr.classify_tod(BASE + 12 * 3600), "off_peak")
        self.assertEqual(mgr.classify_tod(BASE + 18 * 3600), "evening_rush")
        self.assertEqual(mgr.classify_tod(BASE + 23 * 3600), "night")
        self.assertEqual(mgr.classify_tod(BASE + 27 * 3600), "night")

    def test_model_storage(self):
        mgr = TODManager()
        model = PeriodModel(T_cycle=150, T_red=100, confidence=0.8, sample_count=10)
        mgr.update_model("test_001", BASE + 12 * 3600, model)
        retrieved = mgr.get_model("test_001", BASE + 12 * 3600)
        self.assertIsNotNone(retrieved)
        self.assertAlmostEqual(retrieved.T_cycle, 150)


class TestEndToEnd(unittest.TestCase):
    def _run_simulation(self, n_arrivals=14):
        """Simulate hourly arrivals (all within the 'day' TOD period)."""
        world = _make_world("e2e_001")
        sim = RobotArrivalSimulator(world, random_seed=42)
        observations = []
        for i in range(n_arrivals):
            # Deterministic offsets spread arrivals across cycle phases.
            arrival_time = BASE + 6 * 3600 + i * 3600 + (i * 977) % 1800
            obs = sim.simulate_arrival(
                "e2e_001", arrival_time, f"robot_{i + 1:03d}", misclass_prob=0.02
            )
            for o in obs:
                o.arrival_id = i + 1
            observations.extend(obs)
        return world, sim, observations

    def test_full_pipeline(self):
        world, sim, observations = self._run_simulation()
        self.assertGreater(len(observations), 10)

        pipeline = OnlinePipeline(period_configs=SIMPLE_TOD_PERIODS, world=world)
        result = pipeline.run(observations)

        self.assertEqual(len(result.arrivals), 14)
        self.assertGreater(len(result.transitions), 3)

        # Extracted transitions match the TRUE transitions closely.
        true_transitions = sim.get_true_transitions("e2e_001")
        rg_true = [
            t
            for t in true_transitions
            if t.from_color == SignalColor.RED and t.to_color == SignalColor.GREEN
        ]
        matched = 0
        for gt in rg_true:
            if any(
                abs(ext.timestamp - gt.timestamp) <= 2.0
                for ext in result.transitions
                if ext.from_color == SignalColor.RED and ext.to_color == SignalColor.GREEN
            ):
                matched += 1
        self.assertGreaterEqual(matched / max(1, len(rg_true)), 0.8)

        # Final model predicts colors well NEAR the present.  (Absolute
        # phase cannot be extrapolated over hundreds of cycles: a 0.1%
        # T_cycle error drifts ~8s per 100 cycles.  Robots always predict
        # close to "now", so evaluate there.)
        model = result.tod_manager.get_model("e2e_001", BASE + 21 * 3600)
        self.assertIsNotNone(model)
        errors = 0
        for k in range(120):
            t = BASE + 21 * 3600 + k
            if model.predict_color(t) != world.color_at("e2e_001", t):
                errors += 1
        self.assertLess(errors / 120, 0.3)

    def test_causality_no_future_information(self):
        """Decisions/predictions during an arrival must only use past data."""
        world, _, observations = self._run_simulation()
        pipeline = OnlinePipeline(period_configs=SIMPLE_TOD_PERIODS, world=world)
        result = pipeline.run(observations)

        # The first arrival at the intersection has no trained model yet:
        # countdown predictions for RED-observed seconds must be unavailable
        # (NaN).  (Seconds observed GREEN are trivially 0 - no model needed.)
        first = result.arrivals[0]
        red_preds = [
            p for o, p in zip(first.observations, first.pred_ttg) if o.color == SignalColor.RED
        ]
        self.assertTrue(red_preds)
        self.assertTrue(all(np.isnan(p) for p in red_preds))

        # Later arrivals (same TOD period) have a trained model and finite
        # predictions.  Check the last day arrival that actually waited
        # through red (a countdown is only meaningful on red seconds).
        red_arrivals = [
            r
            for r in result.arrivals
            if r.period == "day" and sum(o.color == SignalColor.RED for o in r.observations) >= 5
        ]
        last = red_arrivals[-1]
        self.assertTrue(np.isfinite(last.ttg_mae))
        self.assertLess(last.ttg_mae, 30.0)

    def test_shared_green_constraint(self):
        """Field constraint: T_green is constant across TOD periods.

        With enough day data but only a handful of night arrivals, the
        night model must still get T_red = T_cycle_night - T_green right,
        because the shared green is pooled across periods.
        """
        cfg = {
            "day": {"T_cycle": 120, "T_red": 80, "T_green": 40, "phase_offset": 0},
            "night": {"T_cycle": 100, "T_red": 60, "T_green": 40, "phase_offset": 0},
        }
        world = SignalWorld(period_configs=SIMPLE_TOD_PERIODS)
        world.add_intersection("e2e_001", cfg)
        sim = RobotArrivalSimulator(world, random_seed=7)

        observations = []
        aid = 0
        # Plenty of day arrivals to pin down the day cycle + shared green.
        for i in range(8):
            aid += 1
            obs = sim.simulate_arrival(
                "e2e_001",
                BASE + 8 * 3600 + i * 1800 + (i * 331) % 600,
                f"r{aid}",
                misclass_prob=0.0,
            )
            for o in obs:
                o.arrival_id = aid
            observations.extend(obs)
        # Only three night arrivals.  Their RG transition gaps are 2700s and
        # 3100s (27x and 31x the night cycle 100, coprime multiples), which
        # uniquely identifies T_cycle=100 among all candidates >= 65s.
        for i, offset in enumerate((0, 2737, 5824)):
            aid += 1
            obs = sim.simulate_arrival(
                "e2e_001",
                BASE + 23 * 3600 + offset,
                f"r{aid}",
                misclass_prob=0.0,
            )
            for o in obs:
                o.arrival_id = aid
            observations.extend(obs)

        pipeline = OnlinePipeline(period_configs=SIMPLE_TOD_PERIODS, world=world)
        result = pipeline.run(observations)

        night_model = result.tod_manager.get_model("e2e_001", BASE + 23 * 3600)
        self.assertIsNotNone(night_model)
        # The shared green must be near 40s even though night contributed
        # almost no samples.
        self.assertAlmostEqual(night_model.T_green, 40, delta=10)
        # And the night red follows: T_red = T_cycle_night - T_green.
        self.assertAlmostEqual(night_model.T_red, 60, delta=12)
        self.assertAlmostEqual(night_model.T_cycle, 100, delta=10)

    def test_shared_green_is_isolated_per_intersection(self):
        """Two intersections learned in the same pipeline run must not leak
        state into each other via shared-green pooling.

        ``OnlinePipeline`` indexes its Bayesian learners by intersection so
        that per-intersection bookkeeping (shared-green pooling, re-storing
        all periods' models) only ever walks that one intersection's
        learners. If that indexing were ever broken and fell back to
        scanning every learner in the city by matching on ``iid`` after
        iterating everything, a bug there could leak intersection B's data
        into intersection A's shared-green vote (or vice versa).

        We check this by comparing a combined run (both intersections
        interleaved through one pipeline) against two solo runs (each
        intersection through its own pipeline): true isolation means the
        combined run's per-intersection models must exactly match the solo
        runs', regardless of how well the estimator itself converges.
        """
        cfg_a = {
            "day": {"T_cycle": 120, "T_red": 80, "T_green": 40, "phase_offset": 0},
            "night": {"T_cycle": 120, "T_red": 80, "T_green": 40, "phase_offset": 0},
        }
        cfg_b = {
            "day": {"T_cycle": 150, "T_red": 80, "T_green": 70, "phase_offset": 0},
            "night": {"T_cycle": 150, "T_red": 80, "T_green": 70, "phase_offset": 0},
        }

        def make_observations(iid, cfg, seed, id_offset):
            # arrival_id must be globally unique: group_observations_by_arrival
            # groups purely by that id, regardless of intersection, so two
            # intersections sharing ids would merge into one arrival group.
            world = SignalWorld(period_configs=SIMPLE_TOD_PERIODS)
            world.add_intersection(iid, cfg)
            sim = RobotArrivalSimulator(world, random_seed=seed)
            observations = []
            for i in range(8):
                obs = sim.simulate_arrival(
                    iid,
                    BASE + 8 * 3600 + i * 1800 + (i * 331) % 600,
                    f"{iid}_r{i}",
                    misclass_prob=0.0,
                )
                for o in obs:
                    o.arrival_id = id_offset + i
                observations.extend(obs)
            return world, observations

        world_a, obs_a = make_observations("iid_a", cfg_a, seed=11, id_offset=0)
        world_b, obs_b = make_observations("iid_b", cfg_b, seed=13, id_offset=1000)

        # Solo runs: each intersection learned entirely on its own.
        solo_a = OnlinePipeline(period_configs=SIMPLE_TOD_PERIODS, world=world_a).run(obs_a)
        solo_b = OnlinePipeline(period_configs=SIMPLE_TOD_PERIODS, world=world_b).run(obs_b)
        solo_model_a = solo_a.tod_manager.get_model("iid_a", BASE + 8 * 3600)
        solo_model_b = solo_b.tod_manager.get_model("iid_b", BASE + 8 * 3600)
        self.assertIsNotNone(solo_model_a)
        self.assertIsNotNone(solo_model_b)

        # Combined run: both intersections interleaved through one pipeline.
        combined_world = SignalWorld(period_configs=SIMPLE_TOD_PERIODS)
        combined_world.add_intersection("iid_a", cfg_a)
        combined_world.add_intersection("iid_b", cfg_b)
        combined_obs = sorted(obs_a + obs_b, key=lambda o: o.timestamp)
        combined = OnlinePipeline(period_configs=SIMPLE_TOD_PERIODS, world=combined_world).run(
            combined_obs
        )
        combined_model_a = combined.tod_manager.get_model("iid_a", BASE + 8 * 3600)
        combined_model_b = combined.tod_manager.get_model("iid_b", BASE + 8 * 3600)

        # Compare every field except ``last_updated`` (a wall-clock stamp
        # that legitimately differs between the two runs).
        fields = ("T_cycle", "T_red", "T_green", "phi_offset", "confidence", "sample_count")
        for name in fields:
            self.assertEqual(
                getattr(combined_model_a, name), getattr(solo_model_a, name), msg=f"iid_a.{name}"
            )
            self.assertEqual(
                getattr(combined_model_b, name), getattr(solo_model_b, name), msg=f"iid_b.{name}"
            )


if __name__ == "__main__":
    unittest.main()
