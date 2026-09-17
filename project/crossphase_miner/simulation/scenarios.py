"""
Seongsu (Seongsu-dong, Seoul) demo scenario.

Defines the four intersections (day/night TOD signal plans with independent
phase offsets) and the daily schedules of five delivery robots.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np

from crossphase_miner.core.tod import SIMPLE_TOD_PERIODS
from crossphase_miner.simulation.world import SignalWorld

INTERSECTION_IDS = [
    "seongsu_station",
    "seongsuiero",
    "ttukseom_ro",
    "yeonmujang_gil",
]


def _tod_cfg(
    T_cycle: int,
    T_red: int,
    T_green: int,
    phase_offset: int = 0,
) -> dict:
    """Build a TOD period cycle config with an explicit phase offset."""
    return {
        "T_cycle": T_cycle,
        "T_red": T_red,
        "T_green": T_green,
        # phase_offset: signal phase position (seconds within cycle) at KST
        # midnight; each TOD plan has an independent phase reference.
        "phase_offset": phase_offset,
    }


def build_seongsu_world() -> SignalWorld:
    """
    The four main Seongsu intersections with day/night TOD plans.

    Field observation (intersection 1): the GREEN duration is constant
    across TOD periods; only the RED duration changes:
      slot 1 (20:27~): green 31s | red 119s | cycle 150s
      slot 2 (00:11~): green 31s | red 109s | cycle 140s
    All scenario intersections therefore share one T_green between their
    day and night plans.
    """
    world = SignalWorld(period_configs=SIMPLE_TOD_PERIODS)

    # Seongsu Station - busiest (matches the field observation above)
    world.add_intersection(
        "seongsu_station",
        {
            "day": _tod_cfg(150, 119, 31, phase_offset=0),
            "night": _tod_cfg(140, 109, 31, phase_offset=30),
        },
    )

    # Seongsuiero - medium traffic
    world.add_intersection(
        "seongsuiero",
        {
            "day": _tod_cfg(160, 127, 33, phase_offset=10),
            "night": _tod_cfg(150, 117, 33, phase_offset=40),
        },
    )

    # Ttukseom-ro - less traffic
    world.add_intersection(
        "ttukseom_ro",
        {
            "day": _tod_cfg(150, 122, 28, phase_offset=5),
            "night": _tod_cfg(140, 112, 28, phase_offset=35),
        },
    )

    # Yeonmujang-gil - cafe street entrance, more pedestrians
    world.add_intersection(
        "yeonmujang_gil",
        {
            "day": _tod_cfg(170, 138, 32, phase_offset=12),
            "night": _tod_cfg(160, 128, 32, phase_offset=42),
        },
    )

    return world


def default_base_time() -> float:
    """KST 2024-07-20 00:00:00 as a Unix timestamp (UTC 2024-07-19 15:00)."""
    return datetime(2024, 7, 19, 15, 0, 0, tzinfo=timezone.utc).timestamp()


def generate_robot_schedules(
    num_robots: int = 5,
    base_time: Optional[float] = None,
    seed: int = 2024,
) -> List[Tuple[str, List[Tuple[str, float]]]]:
    """
    Generate one day of arrival schedules per robot.

    Args:
        num_robots: Number of robots.
        base_time: Unix timestamp of KST midnight (default: 2024-07-20).
        seed: Random seed for arrival jitter and intervals.  The default
            keeps experiments reproducible; pass a different seed for a
            different (but still deterministic) day.

    Returns:
        List of (robot_id, [(intersection_id, arrival_timestamp), ...]).
    """
    if base_time is None:
        base_time = default_base_time()

    rng = np.random.RandomState(seed)
    all_schedules: List[Tuple[str, List[Tuple[str, float]]]] = []

    for robot_idx in range(num_robots):
        robot_id = f"robot_{robot_idx + 1:03d}"
        arrivals: List[Tuple[str, float]] = []

        # Each robot has a different shift and arrival frequency.
        if robot_idx == 0:
            work_hours = (6.0, 14.0)  # early shift
            avg_interval = 1800  # 30 min
        elif robot_idx == 1:
            work_hours = (14.0, 22.0)  # late shift
            avg_interval = 1500  # 25 min
        elif robot_idx == 2:
            work_hours = (22.0, 30.0)  # night shift (22:00 - next 06:00)
            avg_interval = 2400  # 40 min
        elif robot_idx == 3:
            work_hours = (8.0, 16.0)  # day shift
            avg_interval = 2000  # ~33 min
        else:
            work_hours = (6.0, 22.0)  # full day, busiest
            avg_interval = 1200  # 20 min

        current_hour = work_hours[0]
        while current_hour < work_hours[1]:
            jitter = rng.uniform(-300, 300)  # ±5 min
            arrival_time = base_time + current_hour * 3600 + jitter

            iid = INTERSECTION_IDS[len(arrivals) % len(INTERSECTION_IDS)]
            arrivals.append((iid, arrival_time))

            interval = avg_interval + rng.uniform(-300, 600)
            current_hour += interval / 3600.0

        all_schedules.append((robot_id, arrivals))

    return all_schedules
