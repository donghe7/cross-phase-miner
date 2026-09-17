"""
Input/output helpers for CrossPhaseMiner pipelines.

Reads and writes the CSV/JSON artifacts produced by the simulation CLI and
consumed by the evaluation CLI.  All functions are pure file-format
adapters; no learning or evaluation logic lives here.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from crossphase_miner.core.models import (
    PhaseTransition,
    SignalColor,
    SignalObservation,
)
from crossphase_miner.core.tod import TODPeriodConfig

_COLOR_MAP: Dict[str, SignalColor] = {
    "RED": SignalColor.RED,
    "GREEN": SignalColor.GREEN,
    "FLASHING_GREEN": SignalColor.FLASHING_GREEN,
    "UNKNOWN": SignalColor.UNKNOWN,
}


def kst_str(ts: float, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """KST wall-clock string for a Unix timestamp (timezone-independent)."""
    return datetime.fromtimestamp(ts + 9 * 3600, tz=timezone.utc).strftime(fmt)


# ────────────────────────────────────────────────
# Loading
# ────────────────────────────────────────────────


def load_observations(csv_path: str) -> List[SignalObservation]:
    """Load 1Hz signal observations from CSV."""
    observations: List[SignalObservation] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            observations.append(
                SignalObservation(
                    arrival_id=int(row["arrival_id"]),
                    robot_id=row["robot_id"],
                    timestamp=float(row["timestamp"]),
                    intersection_id=row["intersection_id"],
                    color=_COLOR_MAP.get(row["color"], SignalColor.UNKNOWN),
                    confidence=float(row["confidence"]),
                )
            )
    return observations


def load_transitions(csv_path: str) -> List[PhaseTransition]:
    """Load pre-computed phase transitions from CSV."""
    transitions: List[PhaseTransition] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            transitions.append(
                PhaseTransition(
                    intersection_id=row["intersection_id"],
                    timestamp=float(row["timestamp"]),
                    from_color=_COLOR_MAP.get(row["from_color"], SignalColor.UNKNOWN),
                    to_color=_COLOR_MAP.get(row["to_color"], SignalColor.UNKNOWN),
                    robot_id=row.get("detected_by", row.get("robot_id", "")),
                )
            )
    return transitions


def load_intersection_configs(
    config_path: str,
) -> Tuple[Dict[str, dict], List[TODPeriodConfig]]:
    """
    Load intersection TOD configurations exported by the simulation CLI.

    Returns:
        (intersections, tod_periods) where intersections maps
        intersection_id -> {"cycle_configs": {period: params}}.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    intersections = raw.get("intersections", raw)
    tod_periods = [
        TODPeriodConfig(
            name=p["name"],
            start_hour=p["start_hour"],
            end_hour=p["end_hour"],
            label=p["label"],
        )
        for p in raw.get("tod_periods", [])
    ]
    return intersections, tod_periods


def group_observations_by_arrival(
    observations: List[SignalObservation],
) -> Dict[int, List[SignalObservation]]:
    """Group observations by arrival_id, each group sorted by timestamp."""
    groups: Dict[int, List[SignalObservation]] = defaultdict(list)
    for obs in observations:
        groups[obs.arrival_id].append(obs)
    for arrival_id in groups:
        groups[arrival_id].sort(key=lambda o: o.timestamp)
    return groups


# ────────────────────────────────────────────────
# Saving
# ────────────────────────────────────────────────


def save_transitions(
    transitions: List[PhaseTransition],
    output_path: str,
) -> None:
    """Save transitions to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "intersection_id",
                "timestamp",
                "datetime_kst",
                "from_color",
                "to_color",
                "robot_id",
            ]
        )
        for t in transitions:
            writer.writerow(
                [
                    t.intersection_id,
                    f"{t.timestamp:.0f}",
                    kst_str(t.timestamp),
                    t.from_color.name,
                    t.to_color.name,
                    t.robot_id,
                ]
            )


def save_observations_csv(
    arrivals: List[dict],
    output_path: str,
) -> None:
    """
    Export simulated arrivals to the observations CSV format.

    ``arrivals`` is a list of dicts with keys: arrival_id, robot_id,
    intersection_id, arrival_time, observations, wait_seconds.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "arrival_id",
                "robot_id",
                "timestamp",
                "datetime_kst",
                "intersection_id",
                "color",
                "confidence",
                "wait_second",
            ]
        )
        for arrival in arrivals:
            for obs in arrival["observations"]:
                writer.writerow(
                    [
                        arrival["arrival_id"],
                        obs.robot_id,
                        f"{obs.timestamp:.1f}",
                        kst_str(obs.timestamp),
                        obs.intersection_id,
                        obs.color.name,
                        f"{obs.confidence:.3f}",
                        f"{obs.timestamp - arrival['arrival_time']:.1f}",
                    ]
                )


def save_true_transitions_csv(
    transitions: List[PhaseTransition],
    output_path: str,
) -> None:
    """Export ground-truth transitions (simulation export format)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "intersection_id",
                "timestamp",
                "datetime_kst",
                "from_color",
                "to_color",
                "detected_by",
            ]
        )
        for t in transitions:
            writer.writerow(
                [
                    t.intersection_id,
                    f"{t.timestamp:.0f}",
                    kst_str(t.timestamp),
                    t.from_color.name,
                    t.to_color.name,
                    t.robot_id,
                ]
            )


def save_intersection_configs(
    world,
    output_path: str,
) -> None:
    """Export a SignalWorld's TOD configurations to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    configs = {
        "tod_periods": [
            {
                "name": p.name,
                "start_hour": p.start_hour,
                "end_hour": p.end_hour,
                "label": p.label,
            }
            for p in world.period_configs
        ],
        "intersections": {
            iid: {"cycle_configs": cycle_configs}
            for iid, cycle_configs in world.intersections.items()
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(configs, f, ensure_ascii=False, indent=2)


def save_summary_json(
    arrivals: List[dict],
    true_transitions: List[PhaseTransition],
    output_path: str,
) -> None:
    """Export per-robot / per-intersection simulation statistics."""
    summary = {
        "region": "Seongsu (성수동), Seoul",
        "date": "2024-07-20",
        "total_arrivals": len(arrivals),
        "robots": {},
        "intersections": {},
    }

    trans_counts: Dict[str, int] = defaultdict(int)
    for t in true_transitions:
        trans_counts[t.intersection_id] += 1

    for arrival in arrivals:
        rid = arrival["robot_id"]
        iid = arrival["intersection_id"]

        robot = summary["robots"].setdefault(
            rid,
            {
                "arrivals": 0,
                "total_wait_seconds": 0,
                "intersections_visited": set(),
            },
        )
        robot["arrivals"] += 1
        robot["total_wait_seconds"] += arrival["wait_seconds"]
        robot["intersections_visited"].add(iid)

        inter = summary["intersections"].setdefault(
            iid,
            {
                "arrivals": 0,
                "total_observations": 0,
                "transitions_detected": trans_counts.get(iid, 0),
            },
        )
        inter["arrivals"] += 1
        inter["total_observations"] += len(arrival["observations"])

    for robot in summary["robots"].values():
        robot["intersections_visited"] = list(robot["intersections_visited"])
        robot["avg_wait_seconds"] = round(robot["total_wait_seconds"] / robot["arrivals"], 1)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
