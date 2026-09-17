"""
Live signal state: what the fleet believes a crossing is showing *right now*.

This is deliberately separate from the learned cycle model.  The model says
what the signal *should* be doing; the live state says what robots can
actually see, and the two are compared in the console.

Two gates are applied, both at 0.70 by default (see ``config``):

* every observation must carry at least ``OBS_CONFIDENCE_THRESHOLD``
  vision confidence, otherwise it is counted as rejected and dropped;
* the winning colour must hold at least ``CONSENSUS_THRESHOLD`` of the
  confidence-weighted votes inside the last ``VOTE_WINDOW_SECONDS``,
  otherwise the crossing is reported as DISPUTED.

A single robot therefore always reaches consensus with itself (one voter,
100% agreement); the gate only bites when robots disagree, which is what
makes a wrong-but-confident detection safe to override.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from crossphase_miner.core.models import SignalColor
from server import config

# Live state labels used on the wire and in the console.
LIVE_OBSERVED = "observed"  # fresh fleet consensus
LIVE_DISPUTED = "disputed"  # robots present but below the quorum
LIVE_PREDICTED = "predicted"  # no robot present, reliable model available
LIVE_UNKNOWN = "unknown"  # no robot, no model


@dataclass
class Vote:
    """The most recent accepted observation from one robot."""

    robot_id: str
    color: SignalColor
    confidence: float
    timestamp: float


@dataclass
class LiveSignalState:
    """Rolling per-crossing view built from robot observations."""

    intersection_id: str
    votes: Dict[str, Vote] = field(default_factory=dict)

    color: SignalColor = SignalColor.UNKNOWN
    agreement: float = 0.0
    voters: int = 0
    updated_at: float = 0.0

    accepted: int = 0
    rejected: int = 0
    disputes: int = 0

    def ingest(
        self,
        robot_id: str,
        color: SignalColor,
        confidence: float,
        timestamp: float,
    ) -> bool:
        """
        Apply the per-observation confidence gate and record the vote.

        Returns True when the observation was accepted.
        """
        if confidence < config.OBS_CONFIDENCE_THRESHOLD:
            self.rejected += 1
            return False

        self.accepted += 1
        previous = self.votes.get(robot_id)
        if previous is None or timestamp >= previous.timestamp:
            self.votes[robot_id] = Vote(robot_id, color, confidence, timestamp)
        if timestamp > self.updated_at:
            self.updated_at = timestamp
        return True

    def resolve(self, now: float) -> Tuple[str, Optional[SignalColor], float, int]:
        """
        Resolve the current fleet view.

        Args:
            now: Current simulated time.

        Returns:
            ``(live_state, color, agreement, voters)``.  ``color`` is None
            when no fresh consensus exists.
        """
        fresh = [v for v in self.votes.values() if now - v.timestamp <= config.VOTE_WINDOW_SECONDS]
        # Drop votes that can never be fresh again to keep the dict small.
        if len(self.votes) > len(fresh) + 4:
            self.votes = {
                v.robot_id: v
                for v in self.votes.values()
                if now - v.timestamp <= config.LIVE_STALE_SECONDS * 4
            }

        if not fresh:
            self.color = SignalColor.UNKNOWN
            self.agreement = 0.0
            self.voters = 0
            return LIVE_UNKNOWN, None, 0.0, 0

        weights: Dict[SignalColor, float] = {}
        for vote in fresh:
            weights[vote.color] = weights.get(vote.color, 0.0) + vote.confidence

        total = sum(weights.values())
        winner, weight = max(weights.items(), key=lambda kv: kv[1])
        agreement = weight / total if total > 0 else 0.0

        self.agreement = agreement
        self.voters = len({v.robot_id for v in fresh})

        if agreement + 1e-9 < config.CONSENSUS_THRESHOLD:
            self.disputes += 1
            self.color = SignalColor.UNKNOWN
            return LIVE_DISPUTED, None, agreement, self.voters

        self.color = winner
        return LIVE_OBSERVED, winner, agreement, self.voters

    def as_dict(self, now: float) -> dict:
        state, color, agreement, voters = self.resolve(now)
        return {
            "state": state,
            "color": color.name if color else None,
            "agreement": round(agreement, 3),
            "voters": voters,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "disputes": self.disputes,
            "age": round(max(0.0, now - self.updated_at), 1),
        }
