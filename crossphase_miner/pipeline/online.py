"""
Online, causal learning pipeline.

Replays robot arrivals in time order.  For each arrival:

1. the CURRENT models (trained only on past data) make per-second
   time-to-green predictions, which are scored against the ground-truth
   world (if provided);
2. the arrival's observations are mined for phase transitions, which update
   the per-(intersection, TOD) learners.

Because learning always happens *after* scoring within an arrival, no future
information ever influences an evaluation point.  Both the Bayesian and the
UKF learner are updated in parallel so their convergence can be compared.

NOTE: the decision fusion engine (per-second crossing decisions) is
not part of this baseline; arrivals use a strict visual confirmation rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from crossphase_miner.core.learners import BayesianPeriodLearner, UKFPeriodLearner
from crossphase_miner.core.models import (
    DecisionResult,
    PeriodModel,
    PhaseTransition,
    SignalColor,
    SignalObservation,
)
from crossphase_miner.core.tod import TODManager, TODPeriodConfig, classify_tod
from crossphase_miner.core.transitions import extract_transitions
from crossphase_miner.pipeline.io import group_observations_by_arrival
from crossphase_miner.simulation.world import SignalWorld


@dataclass
class ArrivalRecord:
    """Everything that happened during one robot arrival."""

    arrival_id: int
    intersection_id: str
    period: str
    observations: List[SignalObservation]
    decisions: List[Tuple[float, DecisionResult]]
    pred_ttg: List[float]  # predicted seconds-to-green per second (NaN = no model)
    gt_ttg: List[float]  # ground-truth seconds-to-green (NaN = unknown)
    ttg_mae: float  # MAE over RED seconds (NaN = not computable)
    transitions: List[PhaseTransition] = field(default_factory=list)


@dataclass
class ModelSnapshot:
    """Learner states after the transitions of one arrival."""

    arrival_index: int  # arrival number within its (intersection, TOD) group
    n_transitions: int
    bayesian: PeriodModel
    ukf: PeriodModel


@dataclass
class PipelineResult:
    """Output of an OnlinePipeline run."""

    arrivals: List[ArrivalRecord]
    transitions: List[PhaseTransition]
    snapshots: Dict[Tuple[str, str], List[ModelSnapshot]]
    tod_manager: TODManager  # final Bayesian models
    ukf_tod_manager: TODManager  # final UKF models
    period_configs: List[TODPeriodConfig]


class OnlinePipeline:
    """Causal replay of observation data with online cycle learning."""

    def __init__(
        self,
        period_configs: List[TODPeriodConfig],
        world: Optional[SignalWorld] = None,
        min_duration: float = 2.0,
        safe_crossing_time: float = 8.0,
    ) -> None:
        """
        Args:
            period_configs: TOD period boundaries.
            world: Optional ground-truth world for scoring predictions.
            min_duration: De-bouncing persistence (seconds) for transition
                extraction.
            safe_crossing_time: Crossing time used by the decision engine.
        """
        self.period_configs = period_configs
        self.world = world
        self.min_duration = min_duration
        self.safe_crossing_time = safe_crossing_time

        self.tod_manager = TODManager(period_configs)
        self.ukf_tod_manager = TODManager(period_configs)
        self._bayes_learners: Dict[Tuple[str, str], BayesianPeriodLearner] = {}
        self._ukf_learners: Dict[Tuple[str, str], UKFPeriodLearner] = {}
        self._arrival_counters: Dict[Tuple[str, str], int] = {}

    def run(self, observations: List[SignalObservation]) -> PipelineResult:
        """Replay all arrivals in time order and return the full result."""
        arrivals = group_observations_by_arrival(observations)
        ordered = sorted(
            arrivals.items(),
            key=lambda item: item[1][0].timestamp if item[1] else 0.0,
        )

        records: List[ArrivalRecord] = []
        all_transitions: List[PhaseTransition] = []
        snapshots: Dict[Tuple[str, str], List[ModelSnapshot]] = {}

        for arrival_id, obs_seq in ordered:
            if not obs_seq:
                continue
            record = self._process_arrival(arrival_id, obs_seq, snapshots)
            records.append(record)
            all_transitions.extend(record.transitions)

        return PipelineResult(
            arrivals=records,
            transitions=all_transitions,
            snapshots=snapshots,
            tod_manager=self.tod_manager,
            ukf_tod_manager=self.ukf_tod_manager,
            period_configs=self.period_configs,
        )

    # -- Internals ---------------------------------------------------------

    def _process_arrival(
        self,
        arrival_id: int,
        obs_seq: List[SignalObservation],
        snapshots: Dict[Tuple[str, str], List[ModelSnapshot]],
    ) -> ArrivalRecord:
        iid = obs_seq[0].intersection_id
        period = classify_tod(obs_seq[0].timestamp, self.period_configs)

        decisions: List[Tuple[float, DecisionResult]] = []
        pred_ttg: List[float] = []
        gt_ttg: List[float] = []

        for obs in obs_seq:
            # 2. Score the current model's countdown prediction.  On a
            # model/vision color conflict we trust vision and produce no
            # model countdown (the model is locally wrong at the boundary).
            model = self.tod_manager.get_model(iid, obs.timestamp)
            pred_ttg.append(self._predicted_ttg(model, obs))
            gt_ttg.append(self._ground_truth_ttg(iid, obs.timestamp))

        mae = self._ttg_mae(iid, obs_seq, pred_ttg, gt_ttg)

        # 3. Learn from this arrival's transitions (after deciding).
        transitions = extract_transitions(obs_seq, min_duration=self.min_duration)
        key = (iid, period)
        arrival_index = self._arrival_counters.get(key, 0) + 1
        self._arrival_counters[key] = arrival_index
        for transition in transitions:
            self._learn(iid, period, transition, arrival_index, snapshots)

        return ArrivalRecord(
            arrival_id=arrival_id,
            intersection_id=iid,
            period=period,
            observations=obs_seq,
            decisions=decisions,
            pred_ttg=pred_ttg,
            gt_ttg=gt_ttg,
            ttg_mae=mae,
            transitions=transitions,
        )

    def _learn(
        self,
        iid: str,
        period: str,
        transition: PhaseTransition,
        arrival_index: int,
        snapshots: Dict[Tuple[str, str], List[ModelSnapshot]],
    ) -> None:
        key = (iid, period)
        bayes = self._bayes_learners.setdefault(key, BayesianPeriodLearner())
        ukf = self._ukf_learners.setdefault(key, UKFPeriodLearner())

        bayes.update(transition)
        ukf_model = ukf.update(transition)

        # Shared-green constraint: pool this intersection's per-period green
        # estimates and apply the result back to every period learner.
        self._apply_shared_green(iid)

        # Re-store models for ALL periods of this intersection: the shared
        # T_red may have changed in periods other than the current one.
        profile = self.tod_manager.create_intersection(iid)
        for (i, p), learner in self._bayes_learners.items():
            if i == iid:
                profile.tod_models[p] = learner.get_model()
        bayes_model = bayes.get_model()

        self.ukf_tod_manager.update_model(iid, transition.timestamp, ukf_model)

        snapshots.setdefault(key, []).append(
            ModelSnapshot(
                arrival_index=arrival_index,
                n_transitions=bayes.sample_count,
                bayesian=bayes_model,
                ukf=ukf_model,
            )
        )

    def _apply_shared_green(self, iid: str) -> None:
        """
        Pool per-period green estimates into one shared T_green per
        intersection (inverse-variance weighted), then apply it back to each
        of the intersection's Bayesian learners.
        """
        votes = [
            learner.green_estimate()
            for (i, _), learner in self._bayes_learners.items()
            if i == iid and learner.sample_count > 0
        ]
        if not votes:
            return
        weights = [1.0 / max(var, 1.0) for _, var in votes]
        T_green = sum(g * w for (g, _), w in zip(votes, weights)) / sum(weights)
        for (i, _), learner in self._bayes_learners.items():
            if i == iid:
                learner.apply_shared_green(T_green)

    def _predicted_ttg(self, model: Optional[PeriodModel], obs: SignalObservation) -> float:
        """
        Predicted seconds-to-green.

        - GREEN observed with high confidence: trivially 0 (no model needed).
        - RED observed: the model's countdown; NaN when there is no reliable
          model or the model's predicted color conflicts with the observed
          color (vision is trusted on conflict).
        - Low-confidence frames (typical of vision misclassifications):
          NaN - neither trusted as green nor scored against the model.
        """
        if obs.color == SignalColor.GREEN:
            return 0.0 if obs.confidence >= 0.8 else float("nan")
        if model is None:
            return float("nan")
        predicted_color = model.predict_color(obs.timestamp)
        if predicted_color not in (obs.color, SignalColor.UNKNOWN):
            return float("nan")
        next_green, conf = model.predict_next_green(obs.timestamp)
        if not np.isfinite(next_green) or conf <= 0:
            return float("nan")
        return max(0.0, next_green - obs.timestamp)

    def _ground_truth_ttg(self, iid: str, timestamp: float) -> float:
        """Ground-truth seconds-to-green (NaN when no world available)."""
        if self.world is None:
            return float("nan")
        if self.world.color_at(iid, timestamp) == SignalColor.GREEN:
            return 0.0
        next_green = self.world.next_color_time(iid, timestamp, SignalColor.GREEN)
        if next_green is None:
            return float("nan")
        return max(0.0, next_green - timestamp)

    def _ttg_mae(
        self,
        iid: str,
        obs_seq: List[SignalObservation],
        pred_ttg: List[float],
        gt_ttg: List[float],
    ) -> float:
        """
        MAE of the countdown prediction over seconds where BOTH the
        observation and the ground truth are RED - the only moments a
        countdown is meaningful.  Misclassified frames (observed green
        during true red) are excluded: the model cannot be blamed for
        vision noise.
        """
        if self.world is None:
            return float("nan")
        errors = [
            abs(p - g)
            for obs, p, g in zip(obs_seq, pred_ttg, gt_ttg)
            if np.isfinite(p)
            and np.isfinite(g)
            and obs.color == SignalColor.RED
            and self.world.color_at(iid, obs.timestamp) == SignalColor.RED
        ]
        return float(np.mean(errors)) if errors else float("nan")
