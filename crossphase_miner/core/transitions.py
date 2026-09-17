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

from typing import List

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

    max_gap = max(0.5, min(2.0, 3.0 * dt))
    episode_id = (
        f"{observations[0].robot_id}:{observations[0].arrival_id}:{observations[0].timestamp}"
    )
    transitions: List[PhaseTransition] = []
    confirmed_color = None
    pending_to = None
    pending_start = None
    last_valid = None
    segment_start = None
    red_phase_start = None
    red_is_exact = False

    for curr in observations:
        if curr.confidence < min_confidence or curr.color not in (
            SignalColor.RED,
            SignalColor.GREEN,
        ):
            continue
        if last_valid is None or curr.timestamp - last_valid > max_gap:
            # Missing evidence is not continuous confirmation. Re-initialize after
            # a gap, so a later triple cannot masquerade as a complete cycle.
            confirmed_color = curr.color
            pending_to = None
            segment_start = curr.timestamp
            red_phase_start = curr.timestamp if curr.color == SignalColor.RED else None
            red_is_exact = False
        last_valid = curr.timestamp
        if curr.color == confirmed_color:
            pending_to = None
            continue
        if pending_to != curr.color:
            pending_to = curr.color
            pending_start = curr.timestamp
            continue
        if curr.timestamp - pending_start + dt < min_duration:
            continue
        transitions.append(
            PhaseTransition(
                intersection_id=curr.intersection_id,
                timestamp=pending_start,
                from_color=confirmed_color,
                to_color=pending_to,
                robot_id=curr.robot_id,
                episode_start=red_phase_start if pending_to == SignalColor.GREEN else None,
                exact_red=red_is_exact if pending_to == SignalColor.GREEN else False,
                observed_since=segment_start,
                episode_id=episode_id,
            )
        )
        if pending_to == SignalColor.RED:
            red_phase_start = pending_start
            red_is_exact = True
        confirmed_color = pending_to
        pending_to = None
    return transitions
