"""
Time-of-Day (TOD) period classification and model management.

This module is the single source of truth for TOD period classification:
``classify_tod`` is used by the simulator, the learners, the pipeline and
the evaluation code.  Do not re-implement period-boundary logic elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from crossphase_miner.core.models import IntersectionProfile, PeriodModel
from crossphase_miner.utils.math_utils import local_hour_of_day


@dataclass
class TODPeriodConfig:
    """Configuration for a Time-of-Day period."""

    name: str
    start_hour: float  # e.g., 6.0 for 6:00 AM
    end_hour: float  # e.g., 8.5 for 8:30 AM
    label: str  # e.g., "morning_rush"


# Default Korean TOD configurations (fine-grained, 4 periods)
DEFAULT_TOD_PERIODS = [
    TODPeriodConfig("Morning Rush", 6.0, 8.5, "morning_rush"),
    TODPeriodConfig("Off Peak", 8.5, 16.0, "off_peak"),
    TODPeriodConfig("Evening Rush", 16.0, 21.0, "evening_rush"),
    TODPeriodConfig("Night", 21.0, 6.0, "night"),
]

# Simplified two-period TOD configuration (day / night)
SIMPLE_TOD_PERIODS = [
    TODPeriodConfig("Day", 6.0, 22.0, "day"),
    TODPeriodConfig("Night", 22.0, 6.0, "night"),
]


def classify_tod(timestamp: float, period_configs: List[TODPeriodConfig]) -> str:
    """
    Classify a Unix timestamp into a TOD period label (KST wall clock).

    Handles periods that wrap around midnight (e.g., night 22:00-06:00).

    Args:
        timestamp: Unix timestamp in seconds.
        period_configs: List of TODPeriodConfig defining period boundaries.

    Returns:
        The label of the matching TOD period (e.g., "day").

    Raises:
        ValueError: If no periods are configured.
    """
    if not period_configs:
        raise ValueError("No TOD periods configured.")

    hour_of_day = local_hour_of_day(timestamp)

    for config in period_configs:
        start = config.start_hour
        end = config.end_hour

        if start < end:
            if start <= hour_of_day < end:
                return config.label
        else:
            if hour_of_day >= start or hour_of_day < end:
                return config.label

    # Gaps between configured periods: fall back to the first period.
    return period_configs[0].label


def next_period_boundary(timestamp: float, period_configs: List[TODPeriodConfig]) -> float:
    """
    Return the timestamp at which the current TOD period ends (KST).

    Used by the simulation world to know when the signal plan changes.
    """
    current = classify_tod(timestamp, period_configs)
    end_hour = next(p.end_hour for p in period_configs if p.label == current)
    # Seconds since KST midnight.
    ssm = (timestamp + 9 * 3600.0) % 86400.0
    delta = (end_hour * 3600.0 - ssm) % 86400.0
    if delta <= 0:
        delta += 86400.0
    return timestamp + delta


class TODManager:
    """
    Stores one PeriodModel per (intersection, TOD period).

    Pure storage: classification is delegated to :func:`classify_tod`.
    """

    def __init__(self, period_configs: Optional[List[TODPeriodConfig]] = None) -> None:
        self.period_configs: List[TODPeriodConfig] = period_configs or DEFAULT_TOD_PERIODS
        self.intersections: Dict[str, IntersectionProfile] = {}

    def classify_tod(self, timestamp: float) -> str:
        """Classify a timestamp into this manager's TOD periods."""
        return classify_tod(timestamp, self.period_configs)

    def get_model(self, intersection_id: str, timestamp: float) -> Optional[PeriodModel]:
        """Return the model for the intersection's current TOD period, if any."""
        profile = self.intersections.get(intersection_id)
        if profile is None:
            return None
        return profile.tod_models.get(self.classify_tod(timestamp))

    def update_model(
        self,
        intersection_id: str,
        timestamp: float,
        model: PeriodModel,
    ) -> None:
        """Store ``model`` under the TOD period that ``timestamp`` falls into."""
        if intersection_id not in self.intersections:
            self.create_intersection(intersection_id)

        profile = self.intersections[intersection_id]
        tod_label = self.classify_tod(timestamp)
        profile.tod_models[tod_label] = model
        profile.total_observations += model.sample_count

    def create_intersection(self, intersection_id: str) -> IntersectionProfile:
        """Create (or return) the profile for an intersection."""
        if intersection_id not in self.intersections:
            self.intersections[intersection_id] = IntersectionProfile(
                intersection_id=intersection_id
            )
        return self.intersections[intersection_id]

    def get_profile(self, intersection_id: str) -> Optional[IntersectionProfile]:
        """Return the IntersectionProfile for an intersection, if present."""
        return self.intersections.get(intersection_id)

    def get_all_intersections(self) -> List[str]:
        """Return all registered intersection IDs."""
        return list(self.intersections.keys())
