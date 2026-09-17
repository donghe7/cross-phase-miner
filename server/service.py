"""
The signal service: one object that owns the whole server-side state.

Responsibilities
----------------
* **Clock.** A simulated clock (``speed`` x wall time) so a demo can watch
  a 150-second cycle turn over in a few seconds.  Robots read it from
  ``GET /v1/clock`` and compute simulated time locally.
* **Live state.** Robot observations pass the confidence gate and the
  consensus quorum in :mod:`server.live_state`.
* **Learning.** Arrival reports carry the phase transitions a robot
  extracted at the edge.  They are queued and consumed by a background
  worker, so a 15 ms Bayesian update never blocks an ingest request.
* **Prediction.** ``predict`` answers "what is this crossing showing, and
  how long until green", using the live state first and the learned model
  as a fallback.
* **Snapshots.** A compact per-crossing array for the console.

Ground truth (``SignalWorld``) is held only so the console can show
learned-vs-true numbers.  Nothing in the ingest, learning or prediction
path reads it.
"""

from __future__ import annotations

import hashlib
import json
import math
import queue
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from crossphase_miner.core.learners import BayesianPeriodLearner
from crossphase_miner.core.models import (
    PeriodModel,
    PhaseTransition,
    SignalColor,
)
from crossphase_miner.core.tod import SIMPLE_TOD_PERIODS, classify_tod
from server import config
from server.city import IntersectionMeta, build_city
from server.live_state import (
    LIVE_DISPUTED,
    LIVE_OBSERVED,
    LIVE_PREDICTED,
    LIVE_UNKNOWN,
    LiveSignalState,
)
from server.store import LearnerCache, StateStore

# Compact state codes used in the snapshot payload.
STATE_CODES = {
    "unknown": 0,
    "observed_red": 1,
    "observed_green": 2,
    "predicted_red": 3,
    "predicted_green": 4,
    "disputed": 5,
}


class SimClock:
    """Simulated clock: ``sim = sim_start + (wall - wall_start) * speed``."""

    def __init__(self, sim_start: float, speed: float) -> None:
        self.sim_start = sim_start
        self.wall_start = time.time()
        self.speed = speed
        self._lock = threading.RLock()

    def now(self) -> float:
        with self._lock:
            return self.sim_start + (time.time() - self.wall_start) * self.speed

    def set_speed(self, speed: float) -> dict:
        """Re-anchor the clock at this instant, preserving simulated time."""
        if speed not in (1, 2, 5, 10, 20):
            raise ValueError("speed must be one of 1, 2, 5, 10, 20")
        with self._lock:
            wall_now = time.time()
            self.sim_start += (wall_now - self.wall_start) * self.speed
            self.wall_start = wall_now
            self.speed = speed
            return self.as_dict()

    def as_dict(self) -> dict:
        with self._lock:
            return {
                "sim_start": self.sim_start,
                "wall_start": self.wall_start,
                "speed": self.speed,
                "sim_now": self.now(),
            }


@dataclass
class RobotStatus:
    """Last reported position and mode of one robot."""

    robot_id: str
    intersection_id: Optional[str] = None
    mode: str = "idle"
    action: str = "TRAVEL"
    waited: float = 0.0
    arrivals: int = 0
    crossings: int = 0
    scout_arrivals: int = 0
    updated_at: float = 0.0
    saved_seconds: float = 0.0
    origin_id: Optional[str] = None
    destination_id: Optional[str] = None
    travel_started_at: Optional[float] = None
    travel_arrives_at: Optional[float] = None
    observation: Optional[dict] = None

    def as_dict(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "intersection_id": self.intersection_id,
            "mode": self.mode,
            "action": self.action,
            "waited": round(self.waited, 1),
            "arrivals": self.arrivals,
            "crossings": self.crossings,
            "scout_arrivals": self.scout_arrivals,
            "saved_seconds": round(self.saved_seconds, 1),
            "updated_at": self.updated_at,
            "origin_id": self.origin_id,
            "destination_id": self.destination_id,
            "travel_started_at": self.travel_started_at,
            "travel_arrives_at": self.travel_arrives_at,
            "observation": self.observation,
        }


@dataclass
class IngestStats:
    """Counters exposed by ``GET /v1/stats``."""

    observations: int = 0
    accepted: int = 0
    rejected: int = 0
    batches: int = 0
    arrivals: int = 0
    transitions: int = 0
    learn_updates: int = 0
    learn_seconds: float = 0.0
    disputes: int = 0
    recent_obs: List[Tuple[float, int]] = field(default_factory=list)

    def rate(self, window: float = 5.0) -> float:
        """Observations per wall-clock second over the last ``window``."""
        now = time.time()
        self.recent_obs = [(t, n) for t, n in self.recent_obs if now - t <= window]
        if not self.recent_obs:
            return 0.0
        return sum(n for _, n in self.recent_obs) / window


class SignalService:
    """Server-side state for a city of crossings and a fleet of robots."""

    def __init__(
        self,
        num_intersections: int = config.NUM_INTERSECTIONS,
        zone_size: int = config.ZONE_SIZE,
        num_hubs: int = config.NUM_HUBS,
        speed: float = config.CLOCK_SPEED,
        sim_start: Optional[float] = None,
        db_path: str = config.DB_PATH,
        seed: int = config.SEED,
        reset: bool = False,
    ) -> None:
        self.world, metas = build_city(num_intersections, zone_size, num_hubs, seed)
        self.metas: Dict[str, IntersectionMeta] = {m.intersection_id: m for m in metas}
        self.order: List[str] = [m.intersection_id for m in metas]
        self.index: Dict[str, int] = {iid: i for i, iid in enumerate(self.order)}

        self.clock = SimClock(sim_start or config.default_sim_start(), speed)
        self.seed = seed
        self.period_configs = SIMPLE_TOD_PERIODS

        self.store = StateStore(db_path)
        if reset:
            self.store.clear()
        self.store.upsert_intersections(
            [
                (m.intersection_id, m.name, m.district, m.grid_x, m.grid_y, int(m.in_zone))
                for m in metas
            ]
        )
        self.learners = LearnerCache(self.store, config.LEARNER_CACHE_SIZE)
        self.models: Dict[Tuple[str, str], PeriodModel] = self.store.load_models()
        self.model_history = self.store.load_model_history()
        self.visit_totals = self.store.load_visit_totals()
        self.visit_yields = self.store.load_recent_visit_yields()
        self.pending_learning = defaultdict(int)

        self.live: Dict[str, LiveSignalState] = {}
        self.robots: Dict[str, RobotStatus] = {}
        self.stats = IngestStats()

        # Exact red-phase measurements per (crossing, period).  A robot that
        # watched a whole red phase (green -> red -> green) knows T_red
        # exactly; a robot that merely arrived during red only knows a lower
        # bound.  The learner's uniform-bound estimator cannot tell them
        # apart, so the fleet-level distinction is kept here and fed back in
        # through the shared-green constraint.
        self._exact_red: Dict[Tuple[str, str], Deque[float]] = defaultdict(lambda: deque(maxlen=20))
        self._exact_red.update(
            {key: deque(values, maxlen=20) for key, values in self.store.load_exact_red().items()}
        )

        self._lock = threading.RLock()
        self._learn_queue: "queue.Queue[Tuple[str, List[PhaseTransition], List[Tuple[str, float]]]]" = queue.Queue(
            maxsize=10000
        )
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._learn_worker, name="learner", daemon=True)
        self._worker.start()

    # -- lifecycle ---------------------------------------------------------

    def shutdown(self) -> None:
        """Stop the learning worker and persist everything."""
        self._stop.set()
        self._worker.join(timeout=5.0)
        with self._lock:
            self.learners.flush()
            for key, model in self.models.items():
                self.store.save_model(key, model)
        self.store.close()

    # -- ingest ------------------------------------------------------------

    def report_travel(
        self, robot_id: str, destination_id: str, started_at: float, arrives_at: float
    ) -> dict:
        """Report route endpoints; no GPS position is inferred from them."""
        if destination_id not in self.metas:
            raise KeyError(destination_id)
        robot = self.robots.setdefault(robot_id, RobotStatus(robot_id))
        robot.origin_id = robot.intersection_id
        robot.intersection_id = None
        robot.destination_id = destination_id
        robot.travel_started_at = started_at
        robot.travel_arrives_at = arrives_at
        robot.action = "TRAVEL"
        robot.mode = "travel"
        robot.updated_at = started_at
        robot.observation = None
        return robot.as_dict()

    def ingest_batch(
        self,
        robot_id: str,
        intersection_id: str,
        observations: List[Tuple[float, str, float]],
        mode: str = "normal",
        action: str = "WAIT",
    ) -> dict:
        """
        Accept one upload of consecutive observations from a single robot.

        Args:
            robot_id: Reporting robot.
            intersection_id: Crossing being observed.
            observations: ``(timestamp, color_name, confidence)`` tuples.
            mode: ``normal`` or ``scout``.
            action: What the robot is doing (``WAIT``, ``CROSS``, ...).

        Returns:
            Accepted/rejected counts and the resulting fleet view.
        """
        if intersection_id not in self.metas:
            raise KeyError(intersection_id)

        state = self.live.setdefault(intersection_id, LiveSignalState(intersection_id))
        accepted = 0
        latest = 0.0
        for ts, color_name, conf in observations:
            color = SignalColor[color_name]
            if state.ingest(robot_id, color, conf, ts):
                accepted += 1
            latest = max(latest, ts)

        rejected = len(observations) - accepted
        self.stats.observations += len(observations)
        self.stats.accepted += accepted
        self.stats.rejected += rejected
        self.stats.batches += 1
        self.stats.recent_obs.append((time.time(), len(observations)))

        robot = self.robots.setdefault(robot_id, RobotStatus(robot_id))
        robot.intersection_id = intersection_id
        robot.mode = mode
        robot.action = action
        robot.updated_at = latest or self.clock.now()
        robot.destination_id = None
        robot.travel_started_at = None
        robot.travel_arrives_at = None
        if observations:
            ts, color_name, conf = max(observations, key=lambda o: o[0])
            robot.observation = {
                "intersection_id": intersection_id,
                "timestamp": ts,
                "color": color_name,
                "confidence": conf,
                "accepted": conf >= config.OBS_CONFIDENCE_THRESHOLD,
            }

        now = self.clock.now()
        live_state, color, agreement, voters = state.resolve(now)
        return {
            "accepted": accepted,
            "rejected": rejected,
            "live_state": live_state,
            "color": color.name if color else None,
            "agreement": round(agreement, 3),
            "voters": voters,
        }

    def submit_arrival(
        self,
        robot_id: str,
        intersection_id: str,
        arrival_time: float,
        depart_time: float,
        mode: str,
        transitions: List[dict],
        waited: float,
        predicted_wait: Optional[float] = None,
        action: str = "CROSS",
        record_id: Optional[str] = None,
    ) -> dict:
        """
        Record a completed arrival and queue its transitions for learning.

        ``transitions`` are the edge-extracted phase changes: dicts with
        ``timestamp``, ``from_color``, ``to_color`` and optional
        ``episode_start`` (the moment the ending red phase began, which is
        what lets the learner estimate T_red).
        """
        if intersection_id not in self.metas:
            raise KeyError(intersection_id)

        exacts: List[Tuple[str, float]] = []
        for t in transitions:
            if t.get("exact_red") and t.get("episode_start") is not None:
                red = float(t["timestamp"]) - float(t["episode_start"])
                if red > 0:
                    period = classify_tod(float(t["timestamp"]), self.period_configs)
                    exacts.append((period, red))

        parsed: List[PhaseTransition] = [
            PhaseTransition(
                intersection_id=intersection_id,
                timestamp=float(t["timestamp"]),
                from_color=SignalColor[t["from_color"]],
                to_color=SignalColor[t["to_color"]],
                robot_id=robot_id,
                episode_start=(
                    float(t["episode_start"]) if t.get("episode_start") is not None else None
                ),
            )
            for t in transitions
        ]

        # Stable fallback for older clients; explicit IDs are carried by new robots.
        identity = json.dumps([robot_id, intersection_id, arrival_time, depart_time, mode, action])
        by_period = defaultdict(int)
        for transition in parsed:
            by_period[classify_tod(transition.timestamp, self.period_configs)] += 1
        record = {
            "record_id": record_id or hashlib.sha256(identity.encode()).hexdigest(),
            "intersection_id": intersection_id,
            "robot_id": robot_id,
            "arrival_time": arrival_time,
            "depart_time": depart_time,
            "mode": mode,
            "action": action,
            "waited": waited,
            "period": classify_tod(arrival_time, self.period_configs),
            "transitions": len(parsed),
            "recorded_at": time.time(),
            "transitions_by_period": dict(by_period),
        }
        if not self.store.save_visit(record):
            return {"queued": 0, "queue_depth": self._learn_queue.qsize(), "duplicate": True}
        self.visit_yields.setdefault(intersection_id, deque(maxlen=20)).append(record)
        total = self.visit_totals.setdefault(
            intersection_id,
            {"robot_ids": set(), "visits": 0, "scout": 0, "normal": 0, "cross": 0, "timeout": 0},
        )
        total["robot_ids"].add(robot_id)
        total["visits"] += 1
        if mode in ("scout", "normal"):
            total[mode] += 1
        if action in ("CROSS", "TIMEOUT"):
            total[action.lower()] += 1

        robot = self.robots.setdefault(robot_id, RobotStatus(robot_id))
        robot.arrivals += 1
        robot.crossings += int(action == "CROSS")
        robot.waited = waited
        robot.action = action
        robot.mode = mode
        robot.intersection_id = intersection_id
        robot.updated_at = depart_time
        if mode == "scout":
            robot.scout_arrivals += 1
        if predicted_wait is not None:
            # Seconds of vision confirmation the model let the robot skip.
            robot.saved_seconds += max(0.0, config.CONFIRM_SECONDS - 3.0)

        self.stats.arrivals += 1
        self.stats.transitions += len(parsed)

        if parsed:
            self.pending_learning[intersection_id] += 1
            try:
                self._learn_queue.put_nowait((intersection_id, parsed, exacts))
            except queue.Full:
                self.pending_learning[intersection_id] -= 1
                return {
                    "queued": 0,
                    "queue_depth": self._learn_queue.qsize(),
                    "dropped": len(parsed),
                }

        return {"queued": len(parsed), "queue_depth": self._learn_queue.qsize()}

    # -- learning ----------------------------------------------------------

    def _learn_worker(self) -> None:
        while not self._stop.is_set():
            try:
                intersection_id, transitions, exacts = self._learn_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            started = time.perf_counter()
            try:
                self._learn(intersection_id, transitions, exacts)
            finally:
                self.pending_learning[intersection_id] = max(
                    0, self.pending_learning[intersection_id] - 1
                )
                self.stats.learn_seconds += time.perf_counter() - started
                self._learn_queue.task_done()

    def _learn(
        self,
        intersection_id: str,
        transitions: List[PhaseTransition],
        exacts: Optional[List[Tuple[str, float]]] = None,
    ) -> None:
        with self._lock:
            for period, red in exacts or []:
                self._exact_red[(intersection_id, period)].append(red)

            for transition in transitions:
                period = classify_tod(transition.timestamp, self.period_configs)
                key = (intersection_id, period)
                learner = self.learners.get(key)
                learner.update(transition)
                self.stats.learn_updates += 1

            # Shared-green constraint: pool this crossing's per-period green
            # estimates and push the result back into each period learner,
            # so a data-poor period inherits a precise red duration.
            self._apply_shared_green(intersection_id)

            checkpoint = []
            histories = {}
            stamp = {
                "sim_time": self.clock.now(),
                "wall_time": time.time(),
                "robots": sorted({t.robot_id for t in transitions if t.robot_id}),
            }
            for key in self.learners.keys_for(intersection_id):
                learner = self.learners.peek(key)
                if learner is None or learner.sample_count == 0:
                    continue
                model = learner.get_model()
                previous = self.models.get(key)
                history = dict(self.model_history.get(key, {}))
                event = {
                    **stamp,
                    "confidence": round(model.confidence, 3),
                    "samples": model.sample_count,
                }
                if previous is None:
                    history.setdefault("first_model", event)
                if model.is_reliable() and not (previous and previous.is_reliable()):
                    history.setdefault("first_reliable", event)
                history["last_update"] = event
                histories[key] = history
                checkpoint.append((key, model, learner, self._exact_red.get(key, ())))
            self.store.save_learning_state(checkpoint, histories)
            for key, model, _, _ in checkpoint:
                self.models[key] = model
                self.model_history[key] = histories[key]

    def _apply_shared_green(self, intersection_id: str) -> None:
        """
        Pool the crossing's periods into one green duration and push it back.

        Field constraint: T_green is the same in every TOD period, only the
        red duration changes.  A period that has an exact red measurement
        votes with ``T_cycle - T_red_exact`` at high weight; a period with
        only random-arrival bounds votes with the learner's own (much
        looser) green estimate.  Each learner then derives its red duration
        from the pooled green, which is how the night plan inherits a
        precise red from the far better estimated cycle.
        """
        votes: List[Tuple[BayesianPeriodLearner, float, float]] = []
        for key in self.learners.keys_for(intersection_id):
            learner = self.learners.peek(key)
            if learner is None or learner.sample_count == 0:
                continue
            exact = self._exact_red.get(key)
            if exact:
                red = statistics.median(exact)
                votes.append((learner, learner.mu_T_cycle - red, 1.0))
            else:
                green, variance = learner.green_estimate()
                votes.append((learner, green, max(variance, 1.0)))

        if not votes:
            return
        weights = [1.0 / variance for _, _, variance in votes]
        total = sum(weights)
        T_green = sum(g * w for (_, g, _), w in zip(votes, weights)) / total
        for learner, _, _ in votes:
            learner.apply_shared_green(T_green)

    # -- prediction --------------------------------------------------------

    def model_for(self, intersection_id: str, now: float) -> Optional[PeriodModel]:
        """Learned model for the crossing's current TOD period, if any."""
        period = classify_tod(now, self.period_configs)
        return self.models.get((intersection_id, period))

    def predict(self, intersection_id: str, now: Optional[float] = None) -> dict:
        """
        Best available answer for "what is this crossing doing?".

        Live fleet consensus wins when it is fresh; otherwise a reliable
        model is used; otherwise the crossing is UNKNOWN and a robot must
        fall back to the conservative visual rule.
        """
        if intersection_id not in self.metas:
            raise KeyError(intersection_id)
        now = self.clock.now() if now is None else now

        live = self.live.get(intersection_id)
        live_view = (
            live.as_dict(now)
            if live is not None
            else {
                "state": LIVE_UNKNOWN,
                "color": None,
                "agreement": 0.0,
                "voters": 0,
                "accepted": 0,
                "rejected": 0,
                "disputes": 0,
                "age": None,
            }
        )

        model = self.model_for(intersection_id, now)
        reliable = bool(model and model.is_reliable())

        source = live_view["state"]
        color: Optional[str] = live_view["color"]
        if source != LIVE_DISPUTED and (
            source != LIVE_OBSERVED
            or live_view["age"] is None
            or live_view["age"] > config.LIVE_STALE_SECONDS
        ):
            if reliable:
                source = LIVE_PREDICTED
                color = model.predict_color(now).name
            elif source == LIVE_OBSERVED:
                source = LIVE_PREDICTED if reliable else LIVE_UNKNOWN
                color = None

        countdown = None
        remaining = None
        if reliable and source != LIVE_DISPUTED:
            current = SignalColor[color] if color else model.predict_color(now)
            if current == SignalColor.RED:
                countdown = round(model.predict_remaining_time(now, SignalColor.RED), 1)
            else:
                remaining = round(model.predict_remaining_time(now, SignalColor.GREEN), 1)
                countdown = 0.0

        return {
            "intersection_id": intersection_id,
            "name": self.metas[intersection_id].name,
            "period": classify_tod(now, self.period_configs),
            "sim_now": now,
            "source": source,
            "color": color,
            "live": live_view,
            "model": (
                {
                    "T_cycle": round(model.T_cycle, 1),
                    "T_red": round(model.T_red, 1),
                    "T_green": round(model.T_green, 1),
                    "confidence": round(model.confidence, 3),
                    "samples": model.sample_count,
                    "reliable": reliable,
                }
                if model
                else None
            ),
            "seconds_to_green": countdown,
            "green_remaining": remaining,
        }

    def truth(self, intersection_id: str, now: Optional[float] = None) -> dict:
        """Ground truth for the console's learned-vs-true panel (demo only)."""
        now = self.clock.now() if now is None else now
        period = classify_tod(now, self.period_configs)
        cfg = self.world.intersections[intersection_id].get(period, {})
        color = self.world.color_at(intersection_id, now)
        next_green = self.world.next_color_time(intersection_id, now, SignalColor.GREEN)
        return {
            "period": period,
            "color": color.name,
            "T_cycle": cfg.get("T_cycle"),
            "T_red": cfg.get("T_red"),
            "T_green": cfg.get("T_green"),
            "seconds_to_green": (
                round(next_green - now, 1) if next_green and color == SignalColor.RED else 0.0
            ),
        }

    # -- views -------------------------------------------------------------

    def learning_summary(self, intersection_id: str, now: float) -> dict:
        period = classify_tod(now, self.period_configs)
        key = (intersection_id, period)
        model = self.models.get(key)
        totals = self.visit_totals.get(intersection_id, {})
        return {
            "period": period,
            "confidence": round(model.confidence, 3) if model else None,
            "samples": model.sample_count if model else 0,
            "status": "reliable"
            if model and model.is_reliable()
            else "learning"
            if model
            else "unlearned",
            "history": self.model_history.get(key, {}),
            "progress": self.learning_progress(intersection_id, period, model),
            "visits": {
                "robots": len(totals.get("robot_ids", ())),
                **{
                    name: totals.get(name, 0)
                    for name in ("visits", "scout", "normal", "cross", "timeout")
                },
            },
        }

    def learning_progress(
        self, intersection_id: str, period: str, model: Optional[PeriodModel]
    ) -> dict:
        samples = model.sample_count if model else 0
        confidence = model.confidence if model else 0.0
        remaining = max(0, PeriodModel.MIN_SAMPLES - samples)
        yields = [
            record["transitions_by_period"].get(period, 0)
            for record in self.visit_yields.get(intersection_id, ())
            if record["mode"] == "scout"
            and "transitions_by_period" in record
            and (record["period"] == period or period in record["transitions_by_period"])
        ]
        mean = sum(yields) / len(yields) if yields else None
        estimate = None
        basis = "confidence" if remaining == 0 else "planning"
        if model and model.is_reliable():
            estimate, basis = [0, 0], "ready"
        elif remaining:
            if mean is not None:
                if mean > 0:
                    estimate = [math.ceil(remaining / mean)] * 2
                    basis = "history"
                else:
                    basis = "no_transitions"
            else:
                # A complete simulated SCOUT normally spans two RED->GREEN
                # edges and 1-2 GREEN->RED edges; this is a planning assumption.
                estimate = [math.ceil(remaining / 4), math.ceil(remaining / 3)]
        return {
            "samples": samples,
            "required_samples": PeriodModel.MIN_SAMPLES,
            "remaining_samples": remaining,
            "sample_progress": min(1.0, samples / PeriodModel.MIN_SAMPLES),
            "confidence_met": confidence >= PeriodModel.MIN_CONFIDENCE,
            "required_confidence": PeriodModel.MIN_CONFIDENCE,
            "estimated_scout_visits": estimate,
            "estimate_basis": basis,
            "recent_scout_visits": len(yields),
            "mean_transitions_per_scout": round(mean, 2) if mean is not None else None,
            "pending_updates": self.pending_learning[intersection_id],
        }

    def snapshot(self, limit: Optional[int] = None) -> dict:
        """
        Compact city-wide snapshot for the console.

        ``tiles`` is a flat list of ``[state_code, countdown]`` in the same
        order as ``GET /v1/intersections``, which keeps a 1200-crossing
        push around 15 KB.
        """
        now = self.clock.now()
        ids = self.order if limit is None else self.order[:limit]
        tiles: List[list] = []
        counts = {name: 0 for name in STATE_CODES}

        for iid in ids:
            view = self._tile(iid, now)
            tiles.append(view)
            counts_key = next(key for key, code in STATE_CODES.items() if code == view[0])
            counts[counts_key] += 1

        reliable = sum(1 for m in self.models.values() if m.is_reliable())
        return {
            "sim_now": now,
            "period": classify_tod(now, self.period_configs),
            "speed": self.clock.speed,
            "tiles": tiles,
            "learning": {
                iid: self.learning_summary(iid, now)
                for iid in ids
                if iid in self.visit_totals or self.model_for(iid, now) is not None
            },
            "counts": counts,
            "stats": {
                "intersections": len(self.order),
                "zone": sum(1 for m in self.metas.values() if m.in_zone),
                "observations": self.stats.observations,
                "accepted": self.stats.accepted,
                "rejected": self.stats.rejected,
                "obs_per_sec": round(self.stats.rate(), 1),
                "arrivals": self.stats.arrivals,
                "transitions": self.stats.transitions,
                "learn_updates": self.stats.learn_updates,
                "learn_ms_avg": (
                    round(self.stats.learn_seconds / self.stats.learn_updates * 1000, 2)
                    if self.stats.learn_updates
                    else 0.0
                ),
                "queue_depth": self._learn_queue.qsize(),
                "models": len(self.models),
                "models_reliable": reliable,
                "learners_in_memory": len(self.learners),
                "learner_evictions": self.learners.evictions,
                "learner_restores": self.learners.restores,
                "disputes": sum(s.disputes for s in self.live.values()),
            },
            "robots": [r.as_dict() for r in sorted(self.robots.values(), key=lambda r: r.robot_id)],
        }

    def _tile(self, iid: str, now: float) -> list:
        """``[state_code, seconds_to_green]`` for one crossing."""
        live = self.live.get(iid)
        if live is not None:
            state, color, _, _ = live.resolve(now)
            if state == LIVE_OBSERVED and now - live.updated_at <= config.LIVE_STALE_SECONDS:
                model = self.model_for(iid, now)
                countdown = -1.0
                if model and model.is_reliable() and color == SignalColor.RED:
                    countdown = round(model.predict_remaining_time(now, SignalColor.RED), 1)
                code = (
                    STATE_CODES["observed_green"]
                    if color == SignalColor.GREEN
                    else STATE_CODES["observed_red"]
                )
                return [code, countdown]
            if state == LIVE_DISPUTED:
                return [STATE_CODES["disputed"], -1.0]

        model = self.model_for(iid, now)
        if model and model.is_reliable():
            color = model.predict_color(now)
            if color == SignalColor.RED:
                return [
                    STATE_CODES["predicted_red"],
                    round(model.predict_remaining_time(now, SignalColor.RED), 1),
                ]
            return [STATE_CODES["predicted_green"], 0.0]

        return [STATE_CODES["unknown"], -1.0]

    def intersection_list(self) -> List[dict]:
        """Static metadata for every managed crossing, in snapshot order."""
        return [self.metas[iid].as_dict() for iid in self.order]

    def detail(self, intersection_id: str) -> dict:
        """Everything the console shows for one selected crossing."""
        now = self.clock.now()
        prediction = self.predict(intersection_id, now)
        prediction["truth"] = self.truth(intersection_id, now)
        prediction["meta"] = self.metas[intersection_id].as_dict()
        prediction["learning"] = self.learning_summary(intersection_id, now)
        prediction["recent_visits"] = self.store.recent_visits(intersection_id)
        prediction["periods"] = {
            period: {
                "T_cycle": round(model.T_cycle, 1),
                "T_red": round(model.T_red, 1),
                "T_green": round(model.T_green, 1),
                "samples": model.sample_count,
                "confidence": round(model.confidence, 3),
                "reliable": model.is_reliable(),
            }
            for (iid, period), model in self.models.items()
            if iid == intersection_id
        }
        model = self.model_for(intersection_id, now)
        # Console comparison uses the model's own phase, even when live votes
        # disagree with it or with each other. The robot decision stays in predict().
        model_color = model.predict_color(now) if model and model.is_reliable() else None
        prediction["model_prediction"] = {
            "color": model_color.name if model_color else None,
            "seconds_to_green": (
                round(model.predict_remaining_time(now, SignalColor.RED), 1)
                if model_color == SignalColor.RED
                else None
            ),
            "green_remaining": (
                round(model.predict_remaining_time(now, SignalColor.GREEN), 1)
                if model_color == SignalColor.GREEN
                else None
            ),
        }
        period = classify_tod(now, self.period_configs)
        true_cfg = self.world.intersections[intersection_id].get(period, {})
        prediction["phase"] = {
            "model": (round(model.phase_position(now), 2) if model else None),
            "model_cycle": round(model.T_cycle, 2) if model else None,
            "model_red": round(model.T_red, 2) if model else None,
            "truth": round(self.world.phase_position(intersection_id, now), 2),
            "truth_cycle": true_cfg.get("T_cycle"),
            "truth_red": true_cfg.get("T_red"),
        }
        prediction["robots_present"] = [
            r.robot_id
            for r in self.robots.values()
            if r.intersection_id == intersection_id
            and now - r.updated_at <= config.LIVE_STALE_SECONDS
        ]
        return prediction

    def zone_ids(self) -> List[str]:
        """Crossings inside the fleet's delivery zone."""
        return [iid for iid in self.order if self.metas[iid].in_zone]

    def hub_ids(self) -> List[str]:
        """Zone crossings every robot route passes through."""
        return [iid for iid in self.order if self.metas[iid].is_hub]
