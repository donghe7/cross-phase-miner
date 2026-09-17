#!/usr/bin/env python3
"""
CrossPhaseMiner evaluation CLI.

Consumes pre-generated robot observation data (from
``python -m crossphase_miner.cli.simulate``) and runs the full online
pipeline on it:

  1. Load 1Hz observations and ground truth
  2. Replay arrivals causally: decide -> score -> learn (Bayesian + UKF)
  3. Score transition extraction against ground truth
  4. Build convergence curves for both learners
  5. Visualize results

Ground truth is used ONLY for scoring - never to seed learner priors.
All outputs go to --output-dir; the data directory is read-only.

Usage:
    python -m crossphase_miner.cli.evaluate [--data-dir simulation]
                                            [--output-dir evaluation]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List

import numpy as np

from crossphase_miner.pipeline.evaluation import (
    build_world_from_configs,
    compute_arrival_convergence,
    compute_convergence_curves,
    evaluate_transition_extraction,
)
from crossphase_miner.pipeline.io import (
    load_intersection_configs,
    load_observations,
    load_transitions,
    save_transitions,
)
from crossphase_miner.pipeline.online import OnlinePipeline
from crossphase_miner.visualization.plots import Visualizer


def run_visualization(
    records,
    result,
    configs: Dict[str, dict],
    convergence: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
    arrival_convergence,
    output_dir: str,
) -> None:
    """Generate the compact evaluation output: three high-density figures.

    1. ``result_summary.png`` - countdown MAE per arrival for every
       intersection, titled with learned vs. true cycle parameters.
    2. ``learner_convergence.png`` - Bayesian vs. UKF convergence.
    3. ``countdown_examples.png`` - countdown to green (predicted vs. ground
       truth) for three example arrivals, showing the learning progress.
    """
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    viz = Visualizer(figsize=(14, 8))

    cycle_ground_truth = {iid: info.get("cycle_configs", {}) for iid, info in configs.items()}
    # The period with the most arrivals (usually "day").
    period_counts: Dict[str, int] = {}
    for rec in records:
        period_counts[rec.period] = period_counts.get(rec.period, 0) + 1
    main_period = max(period_counts, key=period_counts.get) if period_counts else "day"

    # 1. End-to-end summary: countdown MAE per arrival + learned vs. truth.
    p = os.path.join(output_dir, "result_summary.png")
    viz.plot_mae_summary(
        arrival_convergence,
        result.tod_manager,
        cycle_ground_truth,
        period=main_period,
        title="Countdown Prediction Error vs. Arrival Number",
        save_path=p,
    )
    print(f"  Saved: {p}")

    # 2. Learner comparison: Bayesian vs. UKF convergence.
    p = os.path.join(output_dir, "learner_convergence.png")
    viz.plot_learner_comparison(
        convergence,
        period=main_period,
        title="Learner Convergence: Bayesian (blue) vs. UKF (red)",
        save_path=p,
    )
    print(f"  Saved: {p}")

    # 3. Countdown comparison across example arrivals (first predictable /
    # middle / last) at one intersection - shows the learning progress.
    iid0 = sorted(arrival_convergence.keys())[0] if arrival_convergence else None
    examples = arrival_convergence.get(iid0, {}).get(main_period, (None, None))[1] if iid0 else None
    if examples:
        p = os.path.join(output_dir, "countdown_examples.png")
        viz.plot_countdown_examples(
            examples,
            title=f"Countdown to Green: Learning Progress - {iid0} ({main_period})",
            save_path=p,
        )
        print(f"  Saved: {p}")

    viz.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="CrossPhaseMiner Evaluation")
    parser.add_argument(
        "--data-dir", type=str, default="simulation", help="Directory produced by cli.simulate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation",
        help="Output directory for evaluation results",
    )
    args = parser.parse_args()

    t0 = time.time()

    print("=" * 60)
    print("  CrossPhaseMiner - Signal Cycle Learning")
    print("  Online causal pipeline: decide -> score -> learn")
    print("=" * 60)

    observations_path = os.path.join(args.data_dir, "observations.csv")
    transitions_path = os.path.join(args.data_dir, "groundtruth_transitions.csv")
    configs_path = os.path.join(args.data_dir, "groundtruth_configs.json")

    # 1. Load pre-generated data
    print("\n[1/6] Loading generated data...")
    observations = load_observations(observations_path)
    gt_transitions = load_transitions(transitions_path)
    configs, tod_periods = load_intersection_configs(configs_path)
    print(f"  Observations: {len(observations)}")
    print(f"  Ground-truth transitions: {len(gt_transitions)}")
    print(f"  Intersections: {len(configs)}")
    print(f"  TOD periods:   {[p.label for p in tod_periods]}")

    # 2. Run the online causal pipeline (Bayesian + UKF in parallel)
    print("\n[2/6] Running online pipeline (decide -> score -> learn)...")
    world = build_world_from_configs(configs, tod_periods)
    pipeline = OnlinePipeline(period_configs=tod_periods, world=world)
    result = pipeline.run(observations)
    print(f"  Arrivals replayed: {len(result.arrivals)}")
    print(f"  Extracted transitions: {len(result.transitions)}")

    observed_path = os.path.join(args.output_dir, "observed_transitions.csv")
    save_transitions(result.transitions, observed_path)
    print(f"  Saved transitions observed by robots: {observed_path}")

    # 3. Transition extraction quality (ground truth used only for scoring)
    print("\n[3/6] Transition extraction quality (vs ground truth)...")
    quality = evaluate_transition_extraction(result.transitions, gt_transitions)
    print(f"  Extracted: {quality['extracted']}, Ground truth: {quality['ground_truth']}")
    print(
        f"  TP={quality['true_positives']}, FP={quality['false_positives']}, "
        f"FN={quality['false_negatives']}"
    )
    print(
        f"  Precision={quality['precision']:.3f}, Recall={quality['recall']:.3f}, "
        f"F1={quality['f1']:.3f}"
    )

    # 4. Convergence curves for both learners
    print("\n[4/6] Computing convergence curves (Bayesian vs UKF, ±30min near anchor)...")
    convergence = compute_convergence_curves(result.snapshots, world, observations, tod_periods)
    for learner_name, data in sorted(convergence.items()):
        print(f"  [{learner_name}]")
        for iid, period_map in sorted(data.items()):
            for period, series in sorted(period_map.items()):
                if not series["steps"]:
                    continue
                print(
                    f"    {iid} / {period}: "
                    f"color_err {series['color_error'][0]:.3f} -> "
                    f"{series['color_error'][-1]:.3f}, "
                    f"cycle_rel_err {series['cycle_rel_error'][0]:.3f} -> "
                    f"{series['cycle_rel_error'][-1]:.3f} "
                    f"(arrival {series['steps'][0]}-{series['steps'][-1]})"
                )

    # 5. Arrival-level countdown convergence
    print("\n[5/6] Arrival-level countdown convergence...")
    arrival_convergence = compute_arrival_convergence(result.arrivals)
    for iid, period_map in sorted(arrival_convergence.items()):
        for period, (arr_errors, _) in sorted(period_map.items()):
            finite = [e for _, e in arr_errors if np.isfinite(e)]
            if finite:
                print(
                    f"  {iid} / {period}: arrivals={len(arr_errors)}, "
                    f"first finite MAE={finite[0]:.1f}s, last MAE={finite[-1]:.1f}s"
                )
            else:
                print(
                    f"  {iid} / {period}: arrivals={len(arr_errors)}, no reliable predictions yet"
                )

    # 6. Visualization
    print("\n[6/6] Generating visualizations...")
    run_visualization(
        result.arrivals,
        result,
        configs,
        convergence,
        arrival_convergence,
        args.output_dir,
    )

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Evaluation complete in {elapsed:.1f}s")
    print(f"  Input:  {args.data_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
