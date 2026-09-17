"""Bounded, event-time evidence for independent physical signal transitions.

Fusion assumes robot event clocks are synchronized within MERGE_SECONDS. It
improves timestamp estimates, but multiple cameras do not create extra cycles.
"""

from __future__ import annotations

import math
from statistics import median

from crossphase_miner.core.models import PhaseTransition


class TransitionEvidence:
    MERGE_SECONDS = 0.5
    MAX_EVENTS = 128
    MAX_ROBOTS = 64

    def __init__(self):
        self.events = []
        self.retired = 0
        self.cutoff = None
        self.timing_errors = []

    @property
    def count(self):
        return self.retired + len(self.events)

    @property
    def latest(self):
        return max((e["time"] for e in self.events), default=-math.inf)

    def add(self, transition: PhaseTransition, prediction=None):
        t = float(transition.timestamp)
        direction = (transition.from_color.name, transition.to_color.name)
        if direction not in (("RED", "GREEN"), ("GREEN", "RED")) or not math.isfinite(t):
            return False
        if self.cutoff is not None and t <= self.cutoff + self.MERGE_SECONDS:
            return False  # never count an arbitrarily late replay outside the retained window
        candidates = [
            e
            for e in self.events
            if e["direction"] == list(direction)
            and max(e["high"], t) - min(e["low"], t) <= self.MERGE_SECONDS
        ]
        event = min(candidates, key=lambda e: abs(t - e["time"])) if candidates else None
        is_new = event is None
        if is_new:
            # Only a genuinely future, first-seen event is an out-of-sample check.
            if t > self.latest and prediction is not None and prediction.is_reliable():
                phase = prediction.phi_offset + (prediction.T_red if direction[1] == "GREEN" else 0)
                residual = (
                    t - phase + prediction.T_cycle / 2
                ) % prediction.T_cycle - prediction.T_cycle / 2
                self.timing_errors = (self.timing_errors + [abs(residual)])[-50:]
            event = dict(time=t, low=t, high=t, direction=list(direction), sightings={})
            self.events.append(event)
        robot = transition.robot_id
        if robot not in event["sightings"] and len(event["sightings"]) >= self.MAX_ROBOTS:
            return False
        sighting = dict(
            time=t,
            episode_start=transition.episode_start,
            exact_red=transition.exact_red,
            observed_since=transition.observed_since,
            episode_id=transition.episode_id,
        )
        previous = event["sightings"].get(robot)
        # Replays from one camera cannot weight the fused timestamp repeatedly.
        if previous is None or (t, str(transition.episode_id)) < (
            previous["time"],
            str(previous["episode_id"]),
        ):
            event["sightings"][robot] = sighting
        event["low"] = min(event["low"], t)
        event["high"] = max(event["high"], t)
        event["time"] = median(v["time"] for v in event["sightings"].values())
        self.events.sort(key=lambda e: e["time"])
        while len(self.events) > self.MAX_EVENTS:
            removed = self.events.pop(0)
            self.cutoff = max(self.cutoff or -math.inf, removed["high"])
            self.retired += 1
        return is_new

    def rg_times(self):
        return [e["time"] for e in self.events if e["direction"] == ["RED", "GREEN"]][-50:]

    def red_durations(self, exact):
        values = []
        for event in self.events:
            if event["direction"] != ["RED", "GREEN"]:
                continue
            durations = [
                s["time"] - s["episode_start"]
                for s in event["sightings"].values()
                if s["episode_start"] is not None
                and bool(s["exact_red"]) == exact
                and 0 < s["time"] - s["episode_start"] <= 500
            ]
            if durations:
                values.append(median(durations) if exact else max(durations))
        return values[-50:]

    def cycles(self):
        """A cycle needs three consecutive edges in one continuously observed episode."""
        episodes = {}
        for index, event in enumerate(self.events):
            for robot, sighting in event["sightings"].items():
                if sighting["episode_id"] and sighting["observed_since"] is not None:
                    episodes.setdefault((robot, sighting["episode_id"]), []).append(
                        (index, event["direction"][1], sighting)
                    )
        physical = {}
        for edges in episodes.values():
            edges.sort(key=lambda edge: edge[2]["time"])
            for a, b, c in zip(edges, edges[1:], edges[2:]):
                if (a[1], b[1], c[1]) != ("GREEN", "RED", "GREEN"):
                    continue
                t1, t2, t3 = a[2]["time"], b[2]["time"], c[2]["time"]
                if b[2]["observed_since"] > t1 or c[2]["observed_since"] > t1:
                    continue
                green, red = t2 - t1, t3 - t2
                if min(green, red) < 5 or not 30 <= green + red <= 500:
                    continue
                physical.setdefault((a[0], c[0]), []).append((green + red, red, green))
        return [
            tuple(median(v[i] for v in values) for i in range(3)) for values in physical.values()
        ]

    def to_dict(self):
        return dict(
            events=self.events,
            retired=self.retired,
            cutoff=self.cutoff,
            timing_errors=self.timing_errors,
        )

    @classmethod
    def from_dict(cls, data):
        evidence = cls()
        evidence.events = data.get("events", [])
        evidence.retired = data.get("retired", 0)
        evidence.cutoff = data.get("cutoff")
        evidence.timing_errors = data.get("timing_errors", [])
        return evidence
