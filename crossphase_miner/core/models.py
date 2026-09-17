"""
Core data models for CrossPhaseMiner.
All dataclasses and enums used across the system.

Phase convention (single source of truth)
-----------------------------------------
A signal cycle is ``[RED (T_red) -> GREEN (T_green)]`` and repeats every
``T_cycle = T_red + T_green`` seconds.  ``PeriodModel.phi_offset`` is the
absolute timestamp of a *cycle origin* (the start of a red phase):

    phase(t) = (t - phi_offset) % T_cycle
    RED   if phase(t) < T_red
    GREEN otherwise

Every component that produces or consumes a PeriodModel must use this
convention; no coordinate conversion should appear anywhere else.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Dict, Optional, Tuple


class SignalColor(Enum):
    """Enumeration of possible signal colors."""

    RED = 0
    GREEN = 1
    FLASHING_GREEN = 2
    UNKNOWN = 3


class Action(Enum):
    """Decision actions for the robot."""

    WAIT = "WAIT"
    GO = "GO"
    PREPARE = "PREPARE"
    ABORT = "ABORT"


class RiskLevel(Enum):
    """Risk assessment levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class SignalObservation:
    """A single observation of a traffic signal."""

    intersection_id: str
    timestamp: float
    color: SignalColor
    confidence: float = 1.0
    robot_id: str = "robot_001"
    arrival_id: int = 0

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


@dataclass
class PhaseTransition:
    """A detected transition between signal colors.

    ``timestamp`` is the estimated moment of the color change (first second
    the new color was observed), not the confirmation moment.

    ``episode_start`` is the start of the red phase that this transition
    ends (only set for RED->GREEN transitions): the robot's arrival time
    when it arrived during red, or the observed GREEN->RED transition time
    when the red phase was seen to begin.  ``timestamp - episode_start``
    is therefore either the red time the robot still had to wait, or an
    exact T_red sample; the learners use it to estimate T_red.
    """

    intersection_id: str
    timestamp: float
    from_color: SignalColor
    to_color: SignalColor
    robot_id: str = "robot_001"
    episode_start: Optional[float] = None
    exact_red: bool = False
    observed_since: Optional[float] = None
    episode_id: str = ""


@dataclass
class PeriodModel:
    """
    Learned model for a signal cycle period.

    Uses the module-level phase convention: ``phi_offset`` marks the start
    of a red phase (cycle origin).
    """

    T_cycle: float = 120.0
    T_red: float = 80.0
    T_green: float = 40.0
    phi_offset: float = 0.0
    confidence: float = 0.0
    sample_count: int = 0
    last_updated: float = field(default_factory=time.time)

    evidence_version: int = 2
    complete_cycles: int = 0
    timing_mae: Optional[float] = None
    timing_evaluations: int = 0

    MIN_CONFIDENCE: ClassVar[float] = 0.3
    # Sparse-transition models can be confidently WRONG (a spurious exact
    # fit collapses the posterior variance).  Below this sample count the
    # model stays silent instead of giving wild predictions.
    MIN_SAMPLES: ClassVar[int] = 6

    def is_reliable(self) -> bool:
        """Whether the model is trustworthy enough for prediction."""
        return (
            self.confidence >= self.MIN_CONFIDENCE
            and self.sample_count >= self.MIN_SAMPLES
            and self.T_cycle > 0
        )

    def phase_position(self, timestamp: float) -> float:
        """Position of ``timestamp`` within the cycle, in [0, T_cycle)."""
        return (timestamp - self.phi_offset) % self.T_cycle

    def predict_color(self, timestamp: float) -> SignalColor:
        """Predict the signal color at ``timestamp``."""
        if self.confidence < 0.1 or self.T_cycle <= 0:
            return SignalColor.UNKNOWN
        return SignalColor.RED if self.phase_position(timestamp) < self.T_red else SignalColor.GREEN

    def predict_next_green(self, current_time: float) -> Tuple[float, float]:
        """
        Predict when the next green phase starts.

        Returns:
            Tuple of (predicted_time, confidence).  (inf, 0.0) when the
            model is not reliable enough.
        """
        if not self.is_reliable():
            return float("inf"), 0.0

        phase = self.phase_position(current_time)
        if phase < self.T_red:
            wait = self.T_red - phase
        else:
            wait = (self.T_cycle - phase) + self.T_red

        return current_time + wait, self.confidence

    def predict_remaining_time(self, current_time: float, current_color: SignalColor) -> float:
        """
        Predict remaining seconds in the phase indicated by ``current_color``.

        Returns -1.0 when the model is not reliable enough.
        """
        if self.confidence < self.MIN_CONFIDENCE or self.T_cycle <= 0:
            return -1.0

        phase = self.phase_position(current_time)

        if current_color == SignalColor.RED:
            # Time until this red phase ends (green starts).
            if phase < self.T_red:
                return max(0.0, self.T_red - phase)
            # Model thinks it is green but vision says red: the next red
            # runs from the next cycle origin for a full T_red.
            return max(0.0, (self.T_cycle - phase) + self.T_red)
        elif current_color in (SignalColor.GREEN, SignalColor.FLASHING_GREEN):
            # Time until this green phase ends (red starts).
            if phase >= self.T_red:
                return max(0.0, self.T_cycle - phase)
            return max(0.0, (self.T_red - phase) + self.T_green)
        else:
            return -1.0


@dataclass
class IntersectionProfile:
    """Complete profile for a single intersection."""

    intersection_id: str
    tod_models: Dict[str, PeriodModel] = field(default_factory=dict)
    total_observations: int = 0
    creation_time: float = field(default_factory=time.time)


@dataclass
class DecisionResult:
    """Result of a crossing decision."""

    action: Action
    confidence: float
    reason: str
    visual_confirm_required: int
    predicted_green_time: float
    predicted_green_remaining: float
    risk_level: RiskLevel
