#!/usr/bin/env python3
"""
Seongsu 5-robot 1-day simulation data generation.

Generates 24 hours of 10Hz signal observations for 5 delivery robots at 4
Seongsu intersections.  All times are KST (UTC+9); timestamps are Unix
seconds (UTC).

Output layout (flat, ground-truth files carry a ``groundtruth_`` prefix):
    {output_dir}/
      observations.csv              robot observations (10Hz, noisy)
      groundtruth_configs.json      TRUE TOD signal parameters
      groundtruth_transitions.csv   TRUE phase transition times
      simulation_summary.json       simulation statistics

Usage:
    python -m crossphase_miner.cli.simulate [--robots 5] [--misclass 0.03]
                                            [--output-dir simulation]
"""

from __future__ import annotations

import argparse
import os
from typing import List

from crossphase_miner.pipeline.io import (
    kst_str,
    save_intersection_configs,
    save_observations_csv,
    save_summary_json,
    save_true_transitions_csv,
)
from crossphase_miner.simulation.robot import RobotArrivalSimulator
from crossphase_miner.simulation.scenarios import (
    INTERSECTION_IDS,
    build_seongsu_world,
    default_base_time,
    generate_robot_schedules,
)


def generate_observations(
    sim: RobotArrivalSimulator,
    schedules,
    misclass_prob: float = 0.03,
) -> List[dict]:
    """Run all scheduled arrivals; return arrival dicts with observations."""
    all_observations: List[dict] = []
    arrival_counter = 0

    for robot_id, arrivals in schedules:
        for intersection_id, arrival_time in arrivals:
            arrival_counter += 1
            obs = sim.simulate_arrival(
                intersection_id=intersection_id,
                arrival_time=arrival_time,
                robot_id=robot_id,
                misclass_prob=misclass_prob,
                max_wait_seconds=300,
            )
            all_observations.append(
                {
                    "arrival_id": arrival_counter,
                    "robot_id": robot_id,
                    "intersection_id": intersection_id,
                    "arrival_time": arrival_time,
                    "observations": obs,
                    "wait_seconds": len(obs) / sim.obs_rate_hz,
                }
            )

    return all_observations


def main() -> None:
    parser = argparse.ArgumentParser(description="Seongsu 5-robot 1-day simulation data generation")
    parser.add_argument("--robots", type=int, default=5)
    parser.add_argument("--misclass", type=float, default=0.03)
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed for schedules and observation noise "
        "(default 2024 = reproducible demo data)",
    )
    parser.add_argument("--output-dir", type=str, default="simulation")
    args = parser.parse_args()

    print("=" * 60)
    print("  Seongsu 5-Robot 1-Day Simulation")
    print("=" * 60)

    # 1. World
    print("\n[1/4] Setting up Seongsu intersections...")
    world = build_seongsu_world()
    sim = RobotArrivalSimulator(world, random_seed=args.seed)
    print(f"  Intersections: {', '.join(INTERSECTION_IDS)}")

    # 2. Schedules
    print(f"\n[2/4] Generating schedules for {args.robots} robots (seed={args.seed})...")
    schedules = generate_robot_schedules(args.robots, default_base_time(), seed=args.seed)
    total_arrivals = sum(len(a[1]) for a in schedules)
    print(f"  Total arrivals: {total_arrivals}")
    for robot_id, arrivals in schedules:
        print(f"    {robot_id}: {len(arrivals)} arrivals")

    # 3. Observations
    print(f"\n[3/4] Simulating observations (misclass_prob={args.misclass})...")
    all_obs = generate_observations(sim, schedules, misclass_prob=args.misclass)

    total_obs = sum(len(a["observations"]) for a in all_obs)
    total_wait = sum(a["wait_seconds"] for a in all_obs)
    print(f"  Total observations: {total_obs}")
    print(f"  Average wait: {total_wait / total_arrivals:.1f} s")

    print("\n  Sample data (first 3 arrivals):")
    print(
        f"  {'ID':>4} | {'Robot':>10} | {'Time(KST)':>12} | "
        f"{'Intersection':>14} | {'Color':>5} | {'Wait':>4}"
    )
    print("  " + "-" * 65)
    for arr in all_obs[:3]:
        first = arr["observations"][0]
        print(
            f"  {arr['arrival_id']:>4} | {arr['robot_id']:>10} | "
            f"{kst_str(first.timestamp, '%H:%M:%S'):>12} | "
            f"{arr['intersection_id']:>14} | {first.color.name:>5} | "
            f"{arr['wait_seconds']:>4.0f}s"
        )

    # 4. Export
    print("\n[4/4] Exporting data...")
    os.makedirs(args.output_dir, exist_ok=True)

    save_observations_csv(all_obs, os.path.join(args.output_dir, "observations.csv"))
    true_transitions = sim.get_true_transitions()
    save_true_transitions_csv(
        true_transitions,
        os.path.join(args.output_dir, "groundtruth_transitions.csv"),
    )
    save_intersection_configs(world, os.path.join(args.output_dir, "groundtruth_configs.json"))
    save_summary_json(
        all_obs,
        true_transitions,
        os.path.join(args.output_dir, "simulation_summary.json"),
    )

    print("\n  Transitions per intersection:")
    for iid in INTERSECTION_IDS:
        n = len(sim.get_true_transitions(iid))
        print(f"    {iid:16s}: {n:>3} transitions")

    print(f"\n{'=' * 60}")
    print(f"  Done! Data directory: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
