"""
Evaluation metrics for CrossPhaseMiner.

Ground-truth data is used ONLY here - never to seed learner priors.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from crossphase_miner.core.models import (
    PeriodModel,
    PhaseTransition,
    SignalColor,
    SignalObservation,
)
from crossphase_miner.core.tod import TODPeriodConfig, classify_tod
from crossphase_miner.pipeline.online import ArrivalRecord, ModelSnapshot
from crossphase_miner.simulation.world import SignalWorld


def build_world_from_configs(
    configs: Dict[str, dict],
    tod_periods: List[TODPeriodConfig],
) -> SignalWorld:
    """Reconstruct a ground-truth SignalWorld from exported configs."""
    world = SignalWorld(period_configs=tod_periods)
    for iid, info in configs.items():
        world.add_intersection(iid, info.get("cycle_configs", {}))
    return world


def evaluate_transition_extraction(
    extracted: List[PhaseTransition],
    ground_truth: List[PhaseTransition],
    tolerance_seconds: float = 2.0,
) -> Dict[str, float]:
    """
    Compare extracted transitions against ground truth.

    A ground-truth transition counts as recovered when an extracted
    transition with the same intersection and color-change direction exists
    within ``tolerance_seconds``.
    """

    def _match(gt: PhaseTransition, ext: PhaseTransition) -> bool:
        return (
            gt.intersection_id == ext.intersection_id
            and gt.from_color == ext.from_color
            and gt.to_color == ext.to_color
            and abs(gt.timestamp - ext.timestamp) <= tolerance_seconds
        )

    matched_gt = set()
    matched_ext = set()

    for i, gt in enumerate(ground_truth):
        for j, ext in enumerate(extracted):
            if j in matched_ext:
                continue
            if _match(gt, ext):
                matched_gt.add(i)
                matched_ext.add(j)
                break

    tp = len(matched_gt)
    fp = len(extracted) - len(matched_ext)
    fn = len(ground_truth) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "extracted": len(extracted),
        "ground_truth": len(ground_truth),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# Phase-dependent predictions (color, time-to-green) are only scored within
# this window around the model's phase anchor.  A tiny T_cycle error drifts
# the absolute phase by ΔT per cycle, so evaluating one model over a whole
# day measures extrapolation drift, not prediction quality.  Robots always
# predict near "now", so we evaluate near the anchor.
NEAR_WINDOW_SECONDS = 1800.0  # ±30 min


def _near_timestamps(model: PeriodModel, timestamps: List[float]) -> List[float]:
    """Test timestamps within ±NEAR_WINDOW_SECONDS of the model's anchor."""
    anchor = model.phi_offset
    return [t for t in timestamps if abs(t - anchor) <= NEAR_WINDOW_SECONDS]


def _model_color_error(
    model: PeriodModel,
    world: SignalWorld,
    iid: str,
    timestamps: List[float],
) -> float:
    """Mean color-prediction error near the phase anchor (0 = perfect)."""
    timestamps = _near_timestamps(model, timestamps)
    if model.confidence < 0.1 or not timestamps:
        return 1.0
    errors = 0.0
    for t in timestamps:
        pred = model.predict_color(t)
        gt = world.color_at(iid, t)
        if pred == SignalColor.UNKNOWN:
            errors += 0.5
        else:
            errors += 0.0 if pred == gt else 1.0
    return errors / len(timestamps)


def _model_cycle_rel_error(model: PeriodModel, true_t_cycle: float) -> float:
    """Relative T_cycle estimation error."""
    if model.confidence < 0.1 or model.T_cycle <= 0 or true_t_cycle <= 0:
        return 1.0
    return min(1.0, abs(model.T_cycle - true_t_cycle) / true_t_cycle)


def _model_ttg_mae(
    model: PeriodModel,
    world: SignalWorld,
    iid: str,
    timestamps: List[float],
) -> float:
    """MAE of predicted time-to-green over RED timestamps near the anchor."""
    if model.confidence < 0.1 or model.T_cycle <= 0:
        return float("nan")
    errors: List[float] = []
    for t in _near_timestamps(model, timestamps):
        if world.color_at(iid, t) != SignalColor.RED:
            continue
        next_green, conf = model.predict_next_green(t)
        if not np.isfinite(next_green) or conf <= 0:
            continue
        gt_next = world.next_color_time(iid, t, SignalColor.GREEN)
        if gt_next is None:
            continue
        errors.append(abs(next_green - gt_next))
    return float(np.mean(errors)) if errors else float("nan")


def compute_convergence_curves(
    snapshots: Dict[Tuple[str, str], List[ModelSnapshot]],
    world: SignalWorld,
    observations: List[SignalObservation],
    period_configs: List[TODPeriodConfig],
    max_test_points: int = 200,
) -> Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]:
    """
    Build convergence curves for both learners from pipeline snapshots.

    Phase-dependent metrics (color error, time-to-green MAE) are scored
    only within ±30 min of each model's phase anchor (robot's "near-now"
    regime); the T_cycle error is a parameter comparison and needs no
    window.

    Returns:
        {learner_name: {iid: {period: {"steps", "color_error",
        "cycle_rel_error", "time_to_green_mae"}}}} ready for
        ``Visualizer.plot_learner_comparison``.  ``steps`` are arrival
        numbers within each (intersection, TOD period) group.
    """
    # Test timestamps per (iid, period): subsampled observation times.
    test_ts: Dict[Tuple[str, str], List[float]] = {}
    for obs in observations:
        key = (obs.intersection_id, classify_tod(obs.timestamp, period_configs))
        test_ts.setdefault(key, []).append(obs.timestamp)
    for key, ts in test_ts.items():
        ts.sort()
        step = max(1, len(ts) // max_test_points)
        test_ts[key] = ts[::step]

    def _empty() -> Dict[str, List[float]]:
        return {
            "steps": [],
            "color_error": [],
            "cycle_rel_error": [],
            "time_to_green_mae": [],
        }

    result: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = {
        "bayesian": {},
        "ukf": {},
    }

    for (iid, period), snaps in sorted(snapshots.items()):
        timestamps = test_ts.get((iid, period), [])
        cfg = world.intersections.get(iid, {}).get(period)
        if cfg is None and world.intersections.get(iid):
            cfg = next(iter(world.intersections[iid].values()))
        true_t_cycle = cfg.get("T_cycle", 0.0) if cfg else 0.0

        for name in ("bayesian", "ukf"):
            series = result[name].setdefault(iid, {}).setdefault(period, _empty())
            # Step 0: generic prior (first snapshot state has 1 transition,
            # so step 0 approximates the untrained prior).
            for snap in snaps:
                model = snap.bayesian if name == "bayesian" else snap.ukf
                series["steps"].append(snap.arrival_index)
                series["color_error"].append(_model_color_error(model, world, iid, timestamps))
                series["cycle_rel_error"].append(_model_cycle_rel_error(model, true_t_cycle))
                series["time_to_green_mae"].append(_model_ttg_mae(model, world, iid, timestamps))

    return result


def compute_arrival_convergence(
    records: List[ArrivalRecord],
    max_example_arrivals: int = 3,
) -> Dict[
    str,
    Dict[
        str, Tuple[List[Tuple[int, float]], Dict[int, Tuple[List[int], List[float], List[float]]]]
    ],
]:
    """
    Per (intersection, TOD period) countdown-MAE vs. arrival number.

    Returns:
        {iid: {period: (arrival_errors, example_arrivals)}} where
        arrival_errors = [(arrival_number, mae)] and example_arrivals maps
        arrival_number -> (seconds, pred_ttg, gt_ttg) for curve plotting.
    """
    by_key: Dict[Tuple[str, str], List[ArrivalRecord]] = {}
    for rec in records:
        by_key.setdefault((rec.intersection_id, rec.period), []).append(rec)

    result: Dict[str, Dict[str, Tuple]] = {}
    for (iid, period), recs in sorted(by_key.items()):
        recs.sort(key=lambda r: r.observations[0].timestamp)
        arrival_errors: List[Tuple[int, float]] = []
        examples: Dict[int, Tuple[List[int], List[float], List[float]]] = {}

        n = len(recs)
        # Examples for comparison: the first arrival with a usable
        # prediction, a middle one, and the last one - together they show
        # the learning progress.
        finite_idx = [idx for idx, rec in enumerate(recs, start=1) if np.isfinite(rec.ttg_mae)]
        if finite_idx:
            example_idx = {
                finite_idx[0],
                finite_idx[len(finite_idx) // 2],
                finite_idx[-1],
            }
        else:
            example_idx = {1, n // 2 + 1, n}
        for idx, rec in enumerate(recs, start=1):
            arrival_errors.append((idx, rec.ttg_mae))
            if idx in example_idx and len(examples) < max_example_arrivals:
                t0 = rec.observations[0].timestamp
                secs = [o.timestamp - t0 for o in rec.observations]
                examples[idx] = (secs, list(rec.pred_ttg), list(rec.gt_ttg))

        result.setdefault(iid, {})[period] = (arrival_errors, examples)

    return result
