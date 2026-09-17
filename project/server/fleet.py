"""
Fleet simulator: 20 delivery robots as ordinary HTTP clients.

Each robot runs in its own thread and behaves like real hardware would:

1. it reads the simulated clock once and keeps its own offset;
2. before arriving it asks the server what to expect
   (``GET /v1/predict/{id}``) and picks a mode:
   * **normal** - a reliable cycle model exists, so the robot waits for the
     confirmed green and leaves;
   * **scout** - no reliable model, so the robot deliberately stays for a
     full cycle (green -> red -> green) to measure it exactly;
3. while waiting it samples the signal at ``OBS_RATE_HZ`` with 3%
   misclassification, applies the 0.70 confidence gate to its own frames
   and uploads them in batches;
4. after crossing it extracts phase transitions **on the robot** and
   uploads only those (a handful of numbers) plus the arrival summary.

Step 4 is the important one for scale: the server learns from ~2-4 events
per arrival instead of hundreds of raw frames.  The raw upload in step 3
exists so the console can show live state and so the fleet-consensus gate
has something to work with.

Ground truth lives here, not on the server: a robot's camera is the only
thing that ever touches the real signal.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from crossphase_miner.core.models import SignalColor, SignalObservation
from crossphase_miner.core.transitions import extract_transitions
from crossphase_miner.simulation.world import SignalWorld
from server import config
from server.city import build_city

SCOUT_TIMEOUT_SECONDS = 450.0


# --- tiny HTTP client -----------------------------------------------------


class ServerClient:
    """Minimal blocking JSON client (stdlib only, one per robot thread)."""

    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, path: str) -> dict:
        request = urllib.request.Request(self.base_url + path)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode())

    def post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode())


@dataclass
class Clock:
    """Robot-side view of the server's simulated clock."""

    sim_start: float
    wall_start: float
    speed: float
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _confirmed_time: Optional[float] = field(default=None, repr=False)

    def now(self) -> float:
        with self._lock:
            return self.sim_start + (time.time() - self.wall_start) * self.speed

    def synchronize(self, clock_data: dict) -> None:
        """All robots share this updated server-clock estimate."""
        with self._lock:
            self.sim_start = clock_data["sim_now"]
            self.wall_start = time.time()
            self.speed = clock_data["speed"]
            self._confirmed_time = clock_data["sim_now"]

    def sync_interval(self) -> float:
        """At most half a simulated second between server clock updates."""
        with self._lock:
            return min(0.1, 0.5 / max(1.0, self.speed))

    def confirmed_now(self) -> float:
        with self._lock:
            return self._confirmed_time if self._confirmed_time is not None else self.now()

    def sleep_until(self, sim_target: float) -> None:
        # Short sleeps let an in-flight journey or observation batch react
        # to a speed change, instead of waiting out the old wall-time delay.
        while True:
            with self._lock:
                remaining = sim_target - (
                    self.sim_start + (time.time() - self.wall_start) * self.speed
                )
                speed = self.speed
                confirmed = self._confirmed_time
            if remaining <= 0 and (confirmed is None or confirmed >= sim_target):
                return
            # A server-confirmed watermark prevents a stale fast-clock estimate
            # from releasing future observations immediately after a slowdown.
            interval = self.sync_interval()
            time.sleep(
                min(interval, remaining / speed) if speed > 0 and remaining > 0 else interval
            )


# --- robot ----------------------------------------------------------------


class MQTTRobotClient(ServerClient):
    def __init__(self, base_url: str, transport) -> None:
        super().__init__(base_url)
        self.transport = transport

    def post(self, path: str, payload: dict) -> dict:
        return self.transport.post(path, payload)


class Robot:
    """One delivery robot: routes, observes, reports."""

    def __init__(
        self,
        robot_id: str,
        client: ServerClient,
        clock: Clock,
        world: SignalWorld,
        route: List[str],
        hubs: List[str],
        seed: int,
        obs_rate_hz: float = config.OBS_RATE_HZ,
        batch_seconds: float = config.UPLOAD_BATCH_SECONDS,
        misclass_prob: float = config.MISCLASS_PROB,
    ) -> None:
        self.robot_id = robot_id
        self.client = client
        self.clock = clock
        self.world = world
        self.route = route
        self.hubs = hubs
        self.rng = np.random.RandomState(seed)
        self.obs_rate_hz = obs_rate_hz
        self.batch_seconds = batch_seconds
        self.misclass_prob = misclass_prob
        self._route_pos = self.rng.randint(0, max(1, len(route)))

    # -- routing --

    def next_intersection(self) -> str:
        """Alternate between hub crossings and the robot's own route."""
        if self.hubs and self.rng.random() < 0.45:
            return self.hubs[self.rng.randint(0, len(self.hubs))]
        self._route_pos = (self._route_pos + 1) % len(self.route)
        return self.route[self._route_pos]

    # -- perception --

    def _observe(self, intersection_id: str, sim_time: float) -> Tuple[SignalColor, float]:
        """One vision frame: true colour with 3% flips and a confidence."""
        true_color = self.world.color_at(intersection_id, sim_time)
        if self.rng.random() < self.misclass_prob:
            flipped = SignalColor.GREEN if true_color == SignalColor.RED else SignalColor.RED
            return flipped, float(self.rng.uniform(0.55, 0.75))
        return true_color, float(self.rng.uniform(0.88, 0.99))

    # -- one arrival --

    def run_arrival(self) -> None:
        intersection_id = self.next_intersection()

        travel = float(self.rng.uniform(*config.TRAVEL_SECONDS_RANGE))
        started_at = self.clock.now()
        arrives_at = started_at + travel
        try:
            self.client.post(
                "/v1/travel",
                {
                    "robot_id": self.robot_id,
                    "destination_id": intersection_id,
                    "started_at": started_at,
                    "arrives_at": arrives_at,
                },
            )
        except (urllib.error.URLError, OSError):
            pass
        self.clock.sleep_until(arrives_at)

        arrival_time = self.clock.now()
        prediction = self._safe_predict(intersection_id)
        model = (prediction or {}).get("model") or {}
        reliable = bool(model.get("reliable"))
        predicted_wait = (prediction or {}).get("seconds_to_green")
        mode = "normal" if reliable else "scout"

        observations: List[SignalObservation] = []
        dt = 1.0 / self.obs_rate_hz
        confirm_streak: Optional[float] = None
        rg_seen = 0
        last_confirmed: Optional[SignalColor] = None
        pending: Optional[Tuple[SignalColor, float]] = None
        deadline = arrival_time + (SCOUT_TIMEOUT_SECONDS if mode == "scout" else 300.0)
        done = False
        sim_cursor = arrival_time

        while not done and sim_cursor < deadline:
            # If transport/scheduling fell behind, catch up in one upload
            # instead of sending a queue of already-expired small batches.
            batch_end = min(
                max(sim_cursor + self.batch_seconds, self.clock.confirmed_now()), deadline
            )
            batch: List[dict] = []

            while sim_cursor < batch_end:
                color, conf = self._observe(intersection_id, sim_cursor)
                batch.append({"t": sim_cursor, "color": color.name, "conf": round(conf, 3)})
                observations.append(
                    SignalObservation(
                        intersection_id=intersection_id,
                        timestamp=sim_cursor,
                        color=color,
                        confidence=conf,
                        robot_id=self.robot_id,
                    )
                )

                # Robot-side de-bounced colour tracking (0.70 gate applied
                # here as well, so the robot's own decisions use the same
                # evidence the server does).
                if conf >= config.OBS_CONFIDENCE_THRESHOLD:
                    if last_confirmed is None:
                        last_confirmed = color
                        pending = None
                    elif color != last_confirmed:
                        if pending is None or pending[0] != color:
                            pending = (color, sim_cursor)
                        elif sim_cursor - pending[1] >= config.TRANSITION_MIN_DURATION:
                            if last_confirmed == SignalColor.RED and color == SignalColor.GREEN:
                                rg_seen += 1
                            last_confirmed = color
                            pending = None
                    else:
                        pending = None

                    if color == SignalColor.GREEN:
                        if confirm_streak is None:
                            confirm_streak = sim_cursor
                    else:
                        confirm_streak = None

                # Crossing rule.
                if mode == "normal":
                    if (
                        confirm_streak is not None
                        and sim_cursor - confirm_streak + dt >= config.CONFIRM_SECONDS
                    ):
                        done = True
                        break
                else:
                    # Scout: stay until a full cycle has been measured, i.e.
                    # a second confirmed RED->GREEN edge.
                    if rg_seen >= 2 and confirm_streak is not None:
                        done = True
                        break

                sim_cursor += dt

            if batch:
                # Upload only after these simulated observations have occurred,
                # including when the operator slows the clock mid-batch.
                self.clock.sleep_until(sim_cursor)
                self._upload(intersection_id, batch, mode, "CROSS" if done else "WAIT")

        self._report(
            intersection_id,
            arrival_time,
            sim_cursor,
            mode,
            observations,
            predicted_wait,
            "CROSS" if done else "TIMEOUT",
        )

    # -- server calls --

    def _safe_predict(self, intersection_id: str) -> Optional[dict]:
        try:
            return self.client.get(f"/v1/predict/{intersection_id}")
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return None

    def _upload(self, intersection_id: str, batch: List[dict], mode: str, action: str) -> None:
        try:
            self.client.post(
                "/v1/observations",
                {
                    "robot_id": self.robot_id,
                    "intersection_id": intersection_id,
                    "mode": mode,
                    "action": action,
                    "observations": batch,
                },
            )
        except (urllib.error.URLError, OSError):
            pass  # a robot that cannot reach the server keeps driving

    def _report(
        self,
        intersection_id: str,
        arrival_time: float,
        depart_time: float,
        mode: str,
        observations: List[SignalObservation],
        predicted_wait: Optional[float],
        action: str = "CROSS",
    ) -> None:
        transitions = extract_transitions(
            observations,
            min_duration=config.TRANSITION_MIN_DURATION,
            min_confidence=config.OBS_CONFIDENCE_THRESHOLD,
        )
        payload = self._transition_payload(transitions, observations, arrival_time)
        try:
            self.client.post(
                "/v1/arrivals",
                {
                    "record_id": uuid.uuid4().hex,
                    "action": action,
                    "robot_id": self.robot_id,
                    "intersection_id": intersection_id,
                    "arrival_time": arrival_time,
                    "depart_time": depart_time,
                    "mode": mode,
                    "waited": depart_time - arrival_time,
                    "predicted_wait": predicted_wait,
                    "transitions": payload,
                },
            )
        except (urllib.error.URLError, OSError):
            pass

    @staticmethod
    def _transition_payload(
        transitions: List,
        observations: List[SignalObservation],
        arrival_time: float,
    ) -> List[dict]:
        """
        Attach ``episode_start`` to every RED->GREEN edge.

        ``episode_start`` is the moment the ending red phase began: an
        observed GREEN->RED edge when the robot saw the red start, else the
        arrival time if the robot arrived during red.  The difference to
        the RED->GREEN edge is what the learner uses to estimate T_red.
        """
        arrived_on_red = bool(observations) and observations[0].color == SignalColor.RED
        last_gr: Optional[float] = None
        payload: List[dict] = []

        for transition in transitions:
            episode_start = None
            if (
                transition.from_color == SignalColor.RED
                and transition.to_color == SignalColor.GREEN
            ):
                if last_gr is not None:
                    episode_start = last_gr
                elif arrived_on_red:
                    episode_start = arrival_time
            elif (
                transition.from_color == SignalColor.GREEN
                and transition.to_color == SignalColor.RED
            ):
                last_gr = transition.timestamp

            payload.append(
                {
                    "timestamp": transition.timestamp,
                    "from_color": transition.from_color.name,
                    "to_color": transition.to_color.name,
                    "episode_start": episode_start,
                    # An episode that started at an observed GREEN->RED edge
                    # is an exact T_red measurement, not a random-arrival
                    # lower bound.  The server treats the two differently.
                    "exact_red": bool(
                        transition.to_color == SignalColor.GREEN
                        and episode_start is not None
                        and last_gr is not None
                    ),
                }
            )
        return payload


# --- fleet ----------------------------------------------------------------


def build_routes(
    zone_ids: List[str], hub_ids: List[str], num_robots: int, seed: int
) -> Dict[str, List[str]]:
    """Give every robot an overlapping slice of the delivery zone."""
    rng = random.Random(seed)
    non_hub = [iid for iid in zone_ids if iid not in set(hub_ids)]
    routes: Dict[str, List[str]] = {}
    per_robot = max(3, len(non_hub) // max(1, num_robots // 3))

    for index in range(num_robots):
        robot_id = f"robot_{index + 1:03d}"
        if non_hub:
            start = (index * per_robot // 2) % len(non_hub)
            slice_ = [non_hub[(start + offset) % len(non_hub)] for offset in range(per_robot)]
        else:
            slice_ = list(hub_ids)
        rng.shuffle(slice_)
        routes[robot_id] = slice_ or list(hub_ids)
    return routes


def run_fleet(
    base_url: str,
    num_robots: int = config.NUM_ROBOTS,
    duration_seconds: Optional[float] = None,
) -> None:
    """Start ``num_robots`` robot threads against a running server."""
    client = ServerClient(base_url)
    server_config = client.get("/v1/config")

    world, _ = build_city(
        server_config["num_intersections"],
        server_config["zone_size"],
        server_config["num_hubs"],
        server_config["seed"],
    )
    clock_data = client.get("/v1/clock")
    clock = Clock(
        sim_start=clock_data["sim_now"],
        wall_start=time.time(),
        speed=clock_data["speed"],
    )
    clock.synchronize(clock_data)
    zone_ids = server_config["zone_ids"]
    hub_ids = server_config["hub_ids"]
    routes = build_routes(zone_ids, hub_ids, num_robots, server_config["seed"])

    stop = threading.Event()
    threads: List[threading.Thread] = []
    transport = None
    if config.MQTT_ENABLED:
        from server.mqtt_transport import FleetMQTT

        transport = FleetMQTT(clock)

    def sync_clock() -> None:
        sync_client = ServerClient(base_url)
        while not stop.wait(clock.sync_interval()):
            try:
                clock.synchronize(sync_client.get("/v1/clock"))
            except (urllib.error.URLError, OSError, ValueError):
                pass  # sleepers wait for the next confirmed server time

    if transport is None:
        threading.Thread(target=sync_clock, name="clock-sync", daemon=True).start()

    def robot_loop(robot_id: str, route: List[str], seed: int) -> None:
        robot = Robot(
            robot_id=robot_id,
            client=MQTTRobotClient(base_url, transport) if transport else ServerClient(base_url),
            clock=clock,
            world=world,
            route=route,
            hubs=hub_ids,
            seed=seed,
            obs_rate_hz=server_config.get("obs_rate_hz", config.OBS_RATE_HZ),
            batch_seconds=min(
                server_config.get("upload_batch_seconds", config.UPLOAD_BATCH_SECONDS),
                server_config.get("vote_window_seconds", config.VOTE_WINDOW_SECONDS) / 3,
                server_config.get("live_stale_seconds", config.LIVE_STALE_SECONDS) / 3,
            ),
        )
        while not stop.is_set():
            try:
                robot.run_arrival()
            except Exception as error:  # keep the fleet alive
                print(f"[{robot_id}] {type(error).__name__}: {error}")
                time.sleep(1.0)

    for index, (robot_id, route) in enumerate(sorted(routes.items())):
        thread = threading.Thread(
            target=robot_loop,
            args=(robot_id, route, server_config["seed"] + index),
            name=robot_id,
            daemon=True,
        )
        thread.start()
        threads.append(thread)
        time.sleep(0.05)  # stagger starts

    print(
        f"{len(threads)} robots running against {base_url} "
        f"(clock x{clock.speed:.0f}, zone {len(zone_ids)} crossings, "
        f"{len(hub_ids)} hubs). Ctrl-C to stop."
    )
    try:
        if duration_seconds is None:
            while True:
                time.sleep(1.0)
        else:
            time.sleep(duration_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if transport:
            transport.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="CrossPhase robot fleet")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--robots", type=int, default=config.NUM_ROBOTS)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="stop after N wall-clock seconds (default: run forever)",
    )
    args = parser.parse_args()
    run_fleet(args.server, args.robots, args.duration)


if __name__ == "__main__":
    main()
