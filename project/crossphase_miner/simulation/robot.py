"""
Robot observation simulator.

Simulates a delivery robot arriving at an intersection:
- records the signal color at ``obs_rate_hz`` (default 10Hz) while waiting,
- each observation may be misclassified (vision noise),
- the robot crosses after ``confirm_seconds`` consecutive GREEN seconds
  (inline baseline rule; the decision fusion engine is temporarily
  disabled, see crossphase_miner/core/decision.py).

Ground-truth transitions that occurred during the wait window are recorded
separately for evaluation export.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

# NOTE: the decision fusion engine is temporarily disabled
# (see crossphase_miner/core/decision.py).  The robot crosses with an
# inline baseline rule instead: N consecutive seconds of GREEN.
# from crossphase_miner.core.decision import DecisionFusionEngine
from crossphase_miner.core.models import (
    PhaseTransition,
    SignalColor,
    SignalObservation,
)
from crossphase_miner.simulation.world import SignalWorld


class RobotArrivalSimulator:
    """Generates noisy discrete observations of a SignalWorld."""

    def __init__(
        self,
        world: SignalWorld,
        random_seed: int = 42,
        obs_rate_hz: float = 10.0,
        confirm_seconds: float = 5.0,
    ) -> None:
        """
        Args:
            world: Ground-truth signal world.
            random_seed: Seed for the noise generator.
            obs_rate_hz: Observation rate in Hz (default 10).
            confirm_seconds: Consecutive GREEN seconds before crossing
                (inline baseline while the decision engine is disabled).
        """
        self.world = world
        self.rng = np.random.RandomState(random_seed)
        self.obs_rate_hz = float(obs_rate_hz)
        self.confirm_seconds = float(confirm_seconds)
        self._true_transitions: List[PhaseTransition] = []

    def simulate_arrival(
        self,
        intersection_id: str,
        arrival_time: float,
        robot_id: str = "robot_001",
        misclass_prob: float = 0.03,
        max_wait_seconds: int = 300,
    ) -> List[SignalObservation]:
        """
        Simulate one robot arrival; return observations until crossing.

        The robot crosses after ``confirm_seconds`` consecutive GREEN
        seconds, or gives up after ``max_wait_seconds``.
        """
        if intersection_id not in self.world.intersections:
            raise KeyError(f"Intersection '{intersection_id}' not registered.")

        dt = 1.0 / self.obs_rate_hz
        n_steps = int(round(max_wait_seconds * self.obs_rate_hz))
        observations: List[SignalObservation] = []

        # Inline baseline crossing rule (decision engine disabled):
        # track the continuous GREEN streak in seconds.
        green_streak_start: Optional[float] = None

        for step in range(n_steps):
            ts = arrival_time + step * dt
            true_color = self.world.color_at(intersection_id, ts)

            if self.rng.random() < misclass_prob:
                obs_color = SignalColor.GREEN if true_color == SignalColor.RED else SignalColor.RED
                confidence = self.rng.uniform(0.55, 0.75)
            else:
                obs_color = true_color
                confidence = self.rng.uniform(0.88, 0.99)

            obs = SignalObservation(
                intersection_id=intersection_id,
                timestamp=ts,
                color=obs_color,
                confidence=confidence,
                robot_id=robot_id,
            )
            observations.append(obs)

            # --- disabled: decision fusion engine ---
            # self.engine.observe(obs)
            # decision = self.engine.decide(intersection_id, ts)
            # if decision.action == Action.GO:
            #     break
            if obs.color == SignalColor.GREEN:
                if green_streak_start is None:
                    green_streak_start = ts
                if (ts - green_streak_start) + dt >= self.confirm_seconds:
                    break
            else:
                green_streak_start = None

        # Record TRUE transitions that happened during the wait window.
        end_time = arrival_time + len(observations) * dt
        for ts, from_color, to_color in self.world.true_transitions_in_window(
            intersection_id, arrival_time, end_time
        ):
            self._true_transitions.append(
                PhaseTransition(
                    intersection_id=intersection_id,
                    timestamp=ts,
                    from_color=from_color,
                    to_color=to_color,
                    robot_id=robot_id,
                )
            )

        return observations

    def get_true_transitions(self, intersection_id: Optional[str] = None) -> List[PhaseTransition]:
        """TRUE transitions recorded so far (optionally for one intersection)."""
        if intersection_id is None:
            return list(self._true_transitions)
        return [t for t in self._true_transitions if t.intersection_id == intersection_id]

    def clear_history(self) -> None:
        """Drop all recorded true transitions."""
        self._true_transitions.clear()
