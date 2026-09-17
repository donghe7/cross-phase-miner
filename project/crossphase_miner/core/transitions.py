"""
Phase transition extraction from discrete signal observations.

De-bouncing: a color change is only confirmed when the new color persists
for at least ``min_duration`` SECONDS.  Working in time (not sample counts)
keeps the behavior identical at any observation rate (1Hz, 10Hz, ...).

The confirmed transition is timestamped at the *first* sample where the new
color was observed, so the estimate does not lag the true change by the
confirmation delay.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from crossphase_miner.core.models import (
    PhaseTransition,
    SignalColor,
    SignalObservation,
)


def extract_transitions(
    observations: List[SignalObservation],
    min_duration: float = 2.0,
    min_confidence: float = 0.8,
) -> List[PhaseTransition]:
    """
    Extract phase transitions from discrete observations with de-bouncing.

    Algorithm: track the last CONFIRMED color.  When a different color
    appears, watch it; once it has persisted for ``min_duration`` seconds,
    confirm the transition and backdate it to the first sample of the new
    color.  If it reverts before that, ignore the false alarm.

    Observations with confidence below ``min_confidence`` are treated as
    unreliable: they neither start/extend a pending change nor cancel one.
    This filters out the low-confidence misclassifications typical of vision
    models, which would otherwise create spurious transitions in pairs.

    Args:
        observations: List of SignalObservation from a single continuous
            waiting episode (any sampling rate).
        min_duration: Required persistence (seconds) to confirm a change.
        min_confidence: Minimum vision confidence to trust an observation.

    Returns:
        List of confirmed PhaseTransition events.
    """
    if len(observations) < 2:
        return []

    # Median sample interval of this episode (robust to rate changes).
    timestamps = np.array([o.timestamp for o in observations])
    diffs = np.diff(timestamps)
    dt = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0

    episode_start = observations[0].timestamp
    # Start of the red phase that a future RED->GREEN transition ends:
    # the arrival time when the robot arrived during red, or the confirmed
    # GREEN->RED transition time when the red phase was observed to begin
    # (in which case the wait is an EXACT T_red sample).
    red_phase_start: Optional[float] = (
        episode_start if observations[0].color == SignalColor.RED else None
    )
    transitions: List[PhaseTransition] = []
    confirmed_color = observations[0].color
    pending_to: Optional[object] = None
    pending_start_idx: int = 0

    for i in range(1, len(observations)):
        curr = observations[i]

        if curr.confidence < min_confidence:
            # Unreliable frame: ignore it, keep the current state.
            continue

        curr_color = curr.color

        if curr_color == confirmed_color:
            # Back to confirmed color - cancel any pending transition.
            pending_to = None
            continue

        if pending_to is None or curr_color != pending_to:
            # New potential transition.
            pending_to = curr_color
            pending_start_idx = i
        else:
            # Continuing the pending color: confirm once it has persisted
            # for min_duration seconds.
            persisted = (curr.timestamp - observations[pending_start_idx].timestamp) + dt
            if persisted >= min_duration:
                transitions.append(
                    PhaseTransition(
                        intersection_id=curr.intersection_id,
                        timestamp=observations[pending_start_idx].timestamp,
                        from_color=confirmed_color,
                        to_color=pending_to,
                        robot_id=curr.robot_id,
                        episode_start=(
                            red_phase_start
                            if confirmed_color == SignalColor.RED
                            and pending_to == SignalColor.GREEN
                            else None
                        ),
                    )
                )
                if pending_to == SignalColor.RED:
                    # A red phase just started; remember when.
                    red_phase_start = observations[pending_start_idx].timestamp
                confirmed_color = pending_to
                pending_to = None

    return transitions
