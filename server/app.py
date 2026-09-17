"""
HTTP/WebSocket surface of the signal server.

Endpoints
---------
``GET  /``                          the operations console (static page)
``GET  /v1/clock``                  simulated clock, for robot time sync
``POST /v1/clock/speed``            change speed without resetting simulated time
``GET  /v1/config``                 active thresholds and scale settings
``GET  /v1/stats``                  ingest / learning / cache counters
``GET  /v1/intersections``          static metadata for every crossing
``GET  /v1/intersections/{id}``     live state + learned model + truth
``GET  /v1/predict/{id}``           the answer a robot asks for
``POST /v1/observations``           one batch of robot observations
``POST /v1/arrivals``               completed arrival + extracted transitions
``POST /v1/travel``                 route endpoints and estimated arrival time
``GET  /v1/snapshot``               compact city-wide state (polling)
``WS   /ws``                        the same snapshot, pushed

The ingest endpoints are the only hot path: they do O(batch) work and never
touch the learners, which run on a background worker.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from crossphase_miner.core.models import PeriodModel
from server import config
from server.service import SignalService

STATIC_DIR = Path(__file__).parent / "static"

PUSH_INTERVAL_SECONDS = 0.5


# --- request models -------------------------------------------------------


class ObservationIn(BaseModel):
    """One vision frame: ``t`` simulated seconds, colour, confidence."""

    t: float
    color: str
    conf: float = Field(ge=0.0, le=1.0)


class ClockSpeedIn(BaseModel):
    speed: Literal[1, 2, 5, 10, 20]


class ObservationBatch(BaseModel):
    robot_id: str = Field(min_length=1, max_length=config.MAX_ID_LENGTH)
    intersection_id: str = Field(min_length=1, max_length=config.MAX_ID_LENGTH)
    mode: str = "normal"
    action: str = "WAIT"
    observations: List[ObservationIn] = Field(max_length=config.MAX_OBSERVATIONS_PER_BATCH)


class TransitionIn(BaseModel):
    timestamp: float
    from_color: str
    to_color: str
    episode_start: Optional[float] = None
    # True when ``episode_start`` is an observed GREEN->RED edge, i.e. the
    # robot measured a whole red phase instead of only part of one.
    exact_red: bool = False
    observed_since: Optional[float] = None
    episode_id: str = Field(default="", max_length=256)


class TravelReport(BaseModel):
    robot_id: str = Field(min_length=1, max_length=config.MAX_ID_LENGTH)
    destination_id: str = Field(min_length=1, max_length=config.MAX_ID_LENGTH)
    started_at: float
    arrives_at: float


class ArrivalReport(BaseModel):
    robot_id: str = Field(min_length=1, max_length=config.MAX_ID_LENGTH)
    intersection_id: str = Field(min_length=1, max_length=config.MAX_ID_LENGTH)
    arrival_time: float
    depart_time: float
    mode: str = "normal"
    waited: float = 0.0
    predicted_wait: Optional[float] = None
    transitions: List[TransitionIn] = Field(
        default_factory=list, max_length=config.MAX_TRANSITIONS_PER_ARRIVAL
    )
    action: Literal["CROSS", "TIMEOUT"] = "CROSS"
    record_id: Optional[str] = None


def create_app(service: Optional[SignalService] = None) -> FastAPI:
    """Build the FastAPI app around a :class:`SignalService`."""
    app = FastAPI(title="CrossPhase Signal Server", version="1.0")
    app.state.service = service or SignalService()
    app.state.mqtt = None

    @app.on_event("startup")
    async def start_mqtt() -> None:
        if config.MQTT_ENABLED:
            from server.mqtt_transport import ServerMQTT

            app.state.mqtt = ServerMQTT(svc())
            await app.state.mqtt.start()

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def svc() -> SignalService:
        return app.state.service

    @app.get("/")
    async def console() -> FileResponse:
        index = STATIC_DIR / "index.html"
        if not index.exists():
            raise HTTPException(404, "console not installed")
        return FileResponse(str(index))

    @app.get("/v1/clock")
    async def clock() -> dict:
        return svc().clock.as_dict()

    @app.post("/v1/clock/speed")
    async def clock_speed(request: ClockSpeedIn) -> dict:
        return svc().clock.set_speed(request.speed)

    @app.get("/v1/config")
    async def get_config() -> dict:
        service = svc()
        cfg = config.ServerConfig()
        payload = cfg.__dict__.copy()
        payload["zone_ids"] = service.zone_ids()
        payload["hub_ids"] = service.hub_ids()
        payload.update(
            num_intersections=len(service.order),
            zone_size=len(payload["zone_ids"]),
            num_hubs=len(payload["hub_ids"]),
            clock_speed=service.clock.speed,
            seed=service.seed,
        )
        payload["database_backend"] = service.store.backend
        payload["model_min_confidence"] = PeriodModel.MIN_CONFIDENCE
        payload["model_min_samples"] = PeriodModel.MIN_SAMPLES
        payload["observation_transport"] = "mqtt" if config.MQTT_ENABLED else "http"
        return payload

    @app.get("/v1/transport")
    async def transport_status() -> dict:
        return app.state.mqtt.status() if app.state.mqtt else {"enabled": False}

    @app.get("/v1/stats")
    async def stats() -> dict:
        return svc().snapshot(limit=0)["stats"]

    @app.get("/v1/intersections")
    async def intersections() -> dict:
        return {"intersections": svc().intersection_list()}

    @app.get("/v1/intersections/{intersection_id}")
    async def intersection(intersection_id: str) -> dict:
        try:
            return svc().detail(intersection_id)
        except KeyError:
            raise HTTPException(404, f"unknown crossing {intersection_id}")

    @app.get("/v1/predict/{intersection_id}")
    async def predict(intersection_id: str) -> dict:
        try:
            return svc().predict(intersection_id)
        except KeyError:
            raise HTTPException(404, f"unknown crossing {intersection_id}")

    @app.post("/v1/observations")
    async def observations(batch: ObservationBatch) -> dict:
        try:
            return svc().ingest_batch(
                robot_id=batch.robot_id,
                intersection_id=batch.intersection_id,
                observations=[(o.t, o.color, o.conf) for o in batch.observations],
                mode=batch.mode,
                action=batch.action,
            )
        except KeyError:
            raise HTTPException(404, f"unknown crossing {batch.intersection_id}")

    @app.post("/v1/arrivals")
    async def arrivals(report: ArrivalReport) -> dict:
        try:
            return svc().submit_arrival(
                robot_id=report.robot_id,
                intersection_id=report.intersection_id,
                arrival_time=report.arrival_time,
                depart_time=report.depart_time,
                mode=report.mode,
                transitions=[t.model_dump() for t in report.transitions],
                waited=report.waited,
                predicted_wait=report.predicted_wait,
                action=report.action,
                record_id=report.record_id,
            )
        except KeyError:
            raise HTTPException(404, f"unknown crossing {report.intersection_id}")

    @app.post("/v1/travel")
    async def travel(report: TravelReport) -> dict:
        if report.arrives_at < report.started_at:
            raise HTTPException(422, "arrival must not precede departure")
        try:
            return svc().report_travel(**report.model_dump())
        except KeyError:
            raise HTTPException(404, f"unknown crossing {report.destination_id}")

    @app.get("/v1/snapshot")
    async def snapshot() -> dict:
        return svc().snapshot()

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        try:
            while True:
                await socket.send_json(svc().snapshot())
                await asyncio.sleep(PUSH_INTERVAL_SECONDS)
        except WebSocketDisconnect:
            return
        except (RuntimeError, asyncio.CancelledError):
            return

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        if app.state.mqtt:
            await app.state.mqtt.close()
        with contextlib.suppress(Exception):
            svc().shutdown()

    return app


app = None  # built by __main__ / uvicorn factory


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="CrossPhase signal server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--intersections", type=int, default=config.NUM_INTERSECTIONS)
    parser.add_argument("--zone", type=int, default=config.ZONE_SIZE)
    parser.add_argument("--hubs", type=int, default=config.NUM_HUBS)
    parser.add_argument(
        "--speed", type=float, choices=(1, 2, 5, 10, 20), default=config.CLOCK_SPEED
    )
    parser.add_argument(
        "--db",
        default=config.DB_PATH,
        help="SQLite file or PostgreSQL URL; defaults to DATABASE_URL / CP_DB",
    )
    parser.add_argument("--fresh", action="store_true", help="clear CrossPhase tables before start")
    args = parser.parse_args()
    if args.speed not in (1, 2, 5, 10, 20):
        parser.error("speed must be one of 1, 2, 5, 10, 20")

    service = SignalService(
        num_intersections=args.intersections,
        zone_size=args.zone,
        num_hubs=args.hubs,
        speed=args.speed,
        db_path=args.db,
        reset=args.fresh,
    )
    uvicorn.run(create_app(service), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
