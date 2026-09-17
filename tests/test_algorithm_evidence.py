"""Independent event evidence, continuous-cycle measurements and causal scoring."""

import random
import unittest
from dataclasses import replace

from crossphase_miner.core.learners import BayesianPeriodLearner
from crossphase_miner.core.models import PhaseTransition, SignalColor, SignalObservation
from crossphase_miner.core.transitions import extract_transitions
from server.store import learner_from_dict, learner_to_dict, model_from_dict


def cycle_edges(robot="r1", shift=0.0, continuous=True):
    edges = []
    for index, timestamp in enumerate([114, 157, 271, 314, 428, 471]):
        green = index % 2 == 0
        edges.append(
            PhaseTransition(
                "test",
                timestamp + shift,
                SignalColor.RED if green else SignalColor.GREEN,
                SignalColor.GREEN if green else SignalColor.RED,
                robot,
                episode_start=(timestamp - 114 + shift if green else None),
                exact_red=green,
                observed_since=shift if continuous else None,
                episode_id="scout-1",
            )
        )
    return edges


class AlgorithmEvidenceTest(unittest.TestCase):
    def test_three_robots_observing_two_edges_do_not_meet_six_sample_gate(self):
        learner = BayesianPeriodLearner()
        for robot in range(3):
            for edge in cycle_edges(str(robot), shift=robot * 0.1)[:2]:
                learner.update(edge)
        self.assertEqual(learner.sample_count, 2)
        self.assertFalse(learner.get_model().is_reliable())

    def test_complete_cycles_remove_prior_bias_and_duplicate_weight(self):
        learner = BayesianPeriodLearner()
        for edge in cycle_edges():
            learner.update(edge)
        before = learner.get_model()
        for robot, shift in [("r2", -0.1), ("r3", 0.1)]:
            for edge in cycle_edges(robot, shift):
                learner.update(edge)
        model = learner.get_model()
        self.assertEqual(model.sample_count, 6)
        self.assertEqual(model.complete_cycles, 2)
        self.assertAlmostEqual(model.T_cycle, 157)
        self.assertAlmostEqual(model.T_red, 114)
        self.assertAlmostEqual(model.T_green, 43)
        self.assertEqual(model.confidence, before.confidence)
        self.assertTrue(model.is_reliable())

    def test_sparse_or_interrupted_triple_is_not_a_complete_cycle(self):
        for continuous in [False, True]:
            edges = cycle_edges(continuous=continuous)[:3]
            if continuous:
                edges[-1] = replace(edges[-1], observed_since=200)
            learner = BayesianPeriodLearner()
            for edge in edges:
                learner.update(edge)
            self.assertEqual(learner.get_model().complete_cycles, 0)

    def test_same_robot_separate_episodes_do_not_prove_continuity(self):
        learner = BayesianPeriodLearner()
        for i, edge in enumerate(cycle_edges()[:3]):
            learner.update(replace(edge, episode_id=f"visit-{i}"))
        self.assertEqual(learner.get_model().complete_cycles, 0)

    def test_reordered_delivery_preserves_latest_phase_and_estimate(self):
        edges = cycle_edges()
        sequential, shuffled = BayesianPeriodLearner(), BayesianPeriodLearner()
        for edge in edges:
            sequential.update(edge)
        random.Random(42).shuffle(edges)
        for edge in edges:
            shuffled.update(edge)
        expected, actual = sequential.get_model(), shuffled.get_model()
        self.assertEqual(
            (actual.T_cycle, actual.T_red, actual.phi_offset, actual.sample_count),
            (expected.T_cycle, expected.T_red, expected.phi_offset, expected.sample_count),
        )
        shuffled.update(replace(edges[0], timestamp=10, observed_since=None, episode_id="old"))
        self.assertEqual(shuffled._rg_times[-1], 428)

    def test_evidence_and_cycles_survive_restart_without_duplicate_samples(self):
        learner = BayesianPeriodLearner()
        for edge in cycle_edges():
            learner.update(edge)
        restored = learner_from_dict(learner_to_dict(learner))
        for edge in cycle_edges("other"):
            restored.update(edge)
        self.assertEqual(restored.sample_count, 6)
        self.assertEqual(restored.get_model().complete_cycles, 2)
        self.assertEqual(restored.get_model().T_cycle, 157)

    def test_error_is_scored_before_update_and_not_on_duplicate_or_late_event(self):
        learner = BayesianPeriodLearner()
        for edge in cycle_edges():
            learner.update(edge)
        expected = learner.get_model().phi_offset + learner.get_model().T_red + 157
        edge = PhaseTransition(
            "test", expected + 4, SignalColor.RED, SignalColor.GREEN, robot_id="r1"
        )
        learner.update(edge)
        self.assertEqual(learner.get_model().timing_evaluations, 1)
        self.assertAlmostEqual(learner.get_model().timing_mae, 4)
        learner.update(replace(edge, robot_id="r2"))
        learner.update(replace(edge, timestamp=50))
        self.assertEqual(learner.get_model().timing_evaluations, 1)

    def test_gap_and_untrusted_first_frame_do_not_create_false_transitions(self):
        obs = [
            SignalObservation("test", 0, SignalColor.GREEN, 0.1),
            SignalObservation("test", 1, SignalColor.RED, 0.95),
            SignalObservation("test", 2, SignalColor.GREEN, 0.95),
            SignalObservation("test", 12, SignalColor.GREEN, 0.95),
        ]
        self.assertEqual(extract_transitions(obs), [])

    def test_fifteen_fps_extraction_supplies_continuity_and_exact_red(self):
        obs = []
        for frame in range(280 * 15):
            t = frame / 15
            color = SignalColor.RED if t % 157 < 114 else SignalColor.GREEN
            obs.append(SignalObservation("test", t, color, 0.95, "r1"))
        edges = extract_transitions(obs)
        learner = BayesianPeriodLearner()
        for edge in edges:
            learner.update(edge)
        self.assertEqual(len(edges), 3)
        self.assertFalse(edges[0].exact_red)
        self.assertTrue(edges[2].exact_red)
        self.assertEqual(learner.get_model().complete_cycles, 1)
        self.assertAlmostEqual(learner.get_model().T_cycle, 157)

    def test_legacy_counts_are_not_relabelled_as_independent_evidence(self):
        data = learner_to_dict(BayesianPeriodLearner())
        data.pop("evidence")
        data["sample_count"] = 100
        data["rg_times"] = [114, 114.1, 114.2]
        restored = learner_from_dict(data)
        self.assertEqual(restored.sample_count, 0)
        self.assertFalse(restored.get_model().is_reliable())
        model = model_from_dict(dict(T_cycle=157, sample_count=100, confidence=0.99))
        self.assertEqual(model.T_cycle, 157)
        self.assertEqual(model.sample_count, 0)
        self.assertFalse(model.is_reliable())
