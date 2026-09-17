"""
Ground-truth signal world for simulation.

A SignalWorld knows the TRUE state of every registered signal at any time.
It is used to generate synthetic observations and, in evaluation, to score
predictions.  Robots never read the world directly - they only get noisy
observations through the robot simulator.

Cycle configuration per intersection and TOD period::

    {
        "T_cycle": 150,        # seconds
        "T_red": 100,          # seconds
        "T_green": 50,         # seconds
        "phase_offset": 12,    # phase position (seconds within the cycle)
                               # at KST midnight; lets each TOD plan have an
                               # independent phase reference
    }
"""

from __future__ import annotations

from typing import Dict, List, Optional

from crossphase_miner.core.models import SignalColor
from crossphase_miner.core.tod import (
    DEFAULT_TOD_PERIODS,
    TODPeriodConfig,
    classify_tod,
    next_period_boundary,
)


class SignalWorld:
    """Ground-truth state of fixed-cycle signals at multiple intersections."""

    def __init__(
        self,
        period_configs: Optional[List[TODPeriodConfig]] = None,
    ) -> None:
        self.period_configs = period_configs or DEFAULT_TOD_PERIODS
        self.intersections: Dict[str, Dict[str, dict]] = {}

    def add_intersection(
        self,
        intersection_id: str,
        cycle_configs: Dict[str, dict],
    ) -> None:
        """Register an intersection with its per-TOD-period cycle configs."""
        self.intersections[intersection_id] = cycle_configs

    def classify_tod(self, timestamp: float) -> str:
        """TOD period label for a timestamp."""
        return classify_tod(timestamp, self.period_configs)

    def _config_at(self, intersection_id: str, timestamp: float) -> dict:
        """Cycle config in effect at ``timestamp`` (with fallback)."""
        configs = self.intersections[intersection_id]
        tod = classify_tod(timestamp, self.period_configs)
        if tod in configs:
            return configs[tod]
        if configs:
            return next(iter(configs.values()))
        return {"T_cycle": 120, "T_red": 80, "T_green": 40, "phase_offset": 0}

    def phase_position(self, intersection_id: str, timestamp: float) -> float:
        """Position within the current cycle, in [0, T_cycle)."""
        cfg = self._config_at(intersection_id, timestamp)
        seconds_since_midnight = (timestamp + 9 * 3600.0) % 86400.0
        return (seconds_since_midnight + cfg.get("phase_offset", 0.0)) % cfg["T_cycle"]

    def color_at(self, intersection_id: str, timestamp: float) -> SignalColor:
        """The TRUE signal color at ``timestamp``."""
        cfg = self._config_at(intersection_id, timestamp)
        phase = self.phase_position(intersection_id, timestamp)
        return SignalColor.RED if phase < cfg["T_red"] else SignalColor.GREEN

    def next_color_time(
        self,
        intersection_id: str,
        timestamp: float,
        target: SignalColor,
        max_seconds: float = 86400.0,
    ) -> Optional[float]:
        """
        Next time (> timestamp) the signal switches to ``target`` color.

        Analytic within a TOD period; iterates across period boundaries
        (where cycle parameters and phase offsets change).
        """
        t = float(timestamp)
        deadline = timestamp + max_seconds

        for _ in range(64):
            if t > deadline:
                return None
            cfg = self._config_at(intersection_id, t)
            T = cfg["T_cycle"]
            T_red = cfg["T_red"]
            phase = self.phase_position(intersection_id, t)

            if target == SignalColor.GREEN:
                dt = (T_red - phase) % T
            else:  # next switch to RED = next cycle origin
                dt = (T - phase) % T
            if dt <= 1e-9:
                dt = T

            candidate = t + dt
            boundary = next_period_boundary(t, self.period_configs)
            if candidate <= boundary:
                return candidate
            t = boundary

        return None

    def true_transitions_in_window(
        self,
        intersection_id: str,
        start: float,
        end: float,
    ) -> List[tuple]:
        """
        TRUE color changes in ``[start, end)`` as
        ``(timestamp, from_color, to_color)`` tuples.
        """
        transitions: List[tuple] = []
        color = self.color_at(intersection_id, start)
        t = start
        while True:
            other = SignalColor.GREEN if color == SignalColor.RED else SignalColor.RED
            nxt = self.next_color_time(intersection_id, t, other)
            if nxt is None or nxt >= end:
                break
            transitions.append((nxt, color, other))
            color = other
            t = nxt
        return transitions
