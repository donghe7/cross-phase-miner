"""
City-scale signal population for the server demo.

Builds a :class:`SignalWorld` with 1000+ crossings plus the metadata the
server and the console need (display name, district, grid position, and
whether the crossing is inside the fleet's delivery zone).

The four Seongsu crossings measured in the field keep their real cycle
plans and are always the first four entries, so the numbers shown in the
console can still be checked against the field notes.  The remaining
crossings are drawn from the same distribution: one shared green per
crossing across TOD periods, a longer red during the day than at night.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from crossphase_miner.simulation.scenarios import build_seongsu_world
from crossphase_miner.simulation.world import SignalWorld

DISTRICTS = [
    "Seongsu",
    "Ttukseom",
    "Wangsimni",
    "Konkuk",
    "Seoul Forest",
    "Majang",
    "Sindang",
    "Euljiro",
    "Yongsan",
    "Hannam",
]

# The four field-measured crossings, in scenario order.
FIELD_INTERSECTIONS = [
    ("seongsu_station", "Seongsu Station"),
    ("seongsuiero", "Seongsui-ro"),
    ("ttukseom_ro", "Ttukseom-ro"),
    ("yeonmujang_gil", "Yeonmujang-gil"),
]


@dataclass
class IntersectionMeta:
    """Everything about a crossing that is not its signal timing."""

    intersection_id: str
    name: str
    district: str
    grid_x: int
    grid_y: int
    in_zone: bool = False
    is_hub: bool = False
    is_field_site: bool = False

    def as_dict(self) -> dict:
        return {
            "id": self.intersection_id,
            "name": self.name,
            "district": self.district,
            "x": self.grid_x,
            "y": self.grid_y,
            "in_zone": self.in_zone,
            "is_hub": self.is_hub,
            "is_field_site": self.is_field_site,
        }


def build_city(
    num_intersections: int,
    zone_size: int,
    num_hubs: int,
    seed: int = 2024,
) -> Tuple[SignalWorld, List[IntersectionMeta]]:
    """
    Build the ground-truth world and the crossing metadata list.

    Args:
        num_intersections: Total crossings the server will manage.
        zone_size: How many of them are inside the fleet's delivery zone.
        num_hubs: How many zone crossings every robot route passes through.
        seed: Seed for the cycle-plan draw (keeps demos reproducible).

    Returns:
        ``(world, metas)`` where ``metas[i]`` describes the i-th crossing
        and the world holds its true day/night cycle plans.
    """
    world = build_seongsu_world()  # keeps SIMPLE_TOD_PERIODS (day/night)
    rng = np.random.RandomState(seed)

    metas: List[IntersectionMeta] = []
    side = int(np.ceil(np.sqrt(max(num_intersections, 1))))

    for idx in range(num_intersections):
        if idx < len(FIELD_INTERSECTIONS):
            iid, name = FIELD_INTERSECTIONS[idx]
            district = DISTRICTS[0]
            is_field = True
        else:
            iid = f"x{idx:04d}"
            district = DISTRICTS[idx % len(DISTRICTS)]
            name = f"{district} {idx:04d}"
            is_field = False
            world.add_intersection(iid, _random_plan(rng))

        metas.append(
            IntersectionMeta(
                intersection_id=iid,
                name=name,
                district=district,
                grid_x=idx % side,
                grid_y=idx // side,
                is_field_site=is_field,
            )
        )

    # Delivery zone: the field sites plus a contiguous block after them, so
    # the observed crossings form a recognisable cluster in the console.
    zone = metas[: min(zone_size, len(metas))]
    for meta in zone:
        meta.in_zone = True
    for meta in zone[:num_hubs]:
        meta.is_hub = True

    return world, metas


def _random_plan(rng: np.random.RandomState) -> Dict[str, dict]:
    """
    Draw one crossing's day/night plans.

    Follows the field constraint: T_green is identical in both periods and
    only the red duration (hence the cycle) differs.
    """
    T_green = float(rng.randint(24, 40))
    T_cycle_day = float(rng.choice([120, 130, 140, 150, 160, 170, 180]))
    T_cycle_night = T_cycle_day - float(rng.choice([0, 10, 10, 20]))
    return {
        "day": {
            "T_cycle": T_cycle_day,
            "T_red": T_cycle_day - T_green,
            "T_green": T_green,
            "phase_offset": float(rng.randint(0, int(T_cycle_day))),
        },
        "night": {
            "T_cycle": T_cycle_night,
            "T_red": T_cycle_night - T_green,
            "T_green": T_green,
            "phase_offset": float(rng.randint(0, int(T_cycle_night))),
        },
    }
