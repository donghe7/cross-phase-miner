import asyncio
import json
import os
import queue
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
from psycopg import sql

from server import config
from server.fleet import Clock
from server.mqtt_transport import FleetMQTT, ServerMQTT, client
from server.postgres_env import DEFAULT_FILE, database_url
from server.service import SignalService
from server.store import StateStore


async def wait_for(check):
    for _ in range(200):
        if check():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("MQTT condition timed out")


async def check(target):
    config.MQTT_PREFIX = "cp-test/" + uuid.uuid4().hex
    service = SignalService(
        num_intersections=12,
        zone_size=8,
        num_hubs=2,
        speed=0,
        sim_start=config.default_sim_start() + 3600,
        db_path=target,
    )
    bridge = ServerMQTT(service)
    publisher = None
    observer = client("mqtt-check")
    messages = queue.Queue()
    observer.on_connect = lambda c, u, f, r, p: c.subscribe(
        [(config.MQTT_PREFIX + "/replies/check", 0), (config.MQTT_PREFIX + "/crossings/+/state", 0)]
    )
    observer.on_message = lambda c, u, m: messages.put((m.topic, json.loads(m.payload)))
    try:
        await bridge.start()
        observer.connect(config.MQTT_HOST, config.MQTT_PORT)
        observer.loop_start()
        await asyncio.sleep(0.1)
        clock = Clock(service.clock.now(), time.time(), 0)
        publisher = FleetMQTT(clock)
        iid = service.order[0]
        now = service.clock.now()
        publisher.post(
            "/v1/travel",
            dict(robot_id="robot_001", destination_id=iid, started_at=now - 10, arrives_at=now - 1),
        )
        publisher.post(
            "/v1/observations",
            dict(
                robot_id="robot_001",
                intersection_id=iid,
                mode="scout",
                action="WAIT",
                observations=[dict(t=now, color="RED", conf=0.95)],
            ),
        )
        await wait_for(lambda: service.stats.observations == 1)
        assert service.detail(iid)["live"]["color"] == "RED"
        transitions = [
            dict(
                timestamp=now - 600 + i * 60,
                from_color="RED" if i % 2 == 0 else "GREEN",
                to_color="GREEN" if i % 2 == 0 else "RED",
                episode_start=now - 690 + i * 60 if i % 2 == 0 else None,
            )
            for i in range(6)
        ]
        report = dict(
            record_id="visit-check",
            robot_id="robot_001",
            intersection_id=iid,
            arrival_time=now - 900,
            depart_time=now,
            mode="scout",
            action="CROSS",
            waited=900,
            transitions=transitions,
        )
        publisher.post("/v1/arrivals", report)
        publisher.post("/v1/arrivals", report)
        await wait_for(lambda: service.detail(iid)["learning"]["samples"] == 6)
        assert service.detail(iid)["learning"]["visits"]["visits"] == 1
        assert service.detail(iid)["learning"]["status"] == "reliable"
        observer.publish(
            config.MQTT_PREFIX + "/query/check",
            json.dumps(dict(request_id="query-check", intersection_id=iid)),
        )
        replies = []
        states = []

        def received():
            while not messages.empty():
                topic, data = messages.get_nowait()
                (replies if "/replies/" in topic else states).append(data)
            return bool(replies and states)

        await wait_for(received)
        assert replies[-1]["request_id"] == "query-check"
        assert replies[-1]["detail"]["learning"]["samples"] == 6
        assert states[-1]["intersection_id"] == iid
        assert states[-1]["sim_now"] == now
        database = StateStore(target)
        try:
            assert database.load_visit_totals()[iid]["visits"] == 1
            assert database.load_model_history()[(iid, "day")]["first_reliable"]["samples"] == 6
        finally:
            database.close()
        assert bridge.errors == 0
        print(
            "PASS:",
            service.store.backend,
            "MQTT travel, observation, 6-sample learning, QoS1 duplicate arrival, live subscription, correlated query, model/visit persistence",
        )
    finally:
        if publisher:
            publisher.close()
        observer.disconnect()
        observer.loop_stop()
        await bridge.close()
        service.shutdown()


def main():
    with tempfile.TemporaryDirectory(prefix="cp-mqtt-") as folder:
        asyncio.run(check(str(Path(folder) / "test.sqlite3")))

    base = os.environ.get("CP_TEST_POSTGRES_URL") or database_url(DEFAULT_FILE)
    schema = "cp_mqtt_test_" + uuid.uuid4().hex
    with psycopg.connect(base, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        parts = urlsplit(base)
        query = dict(parse_qsl(parts.query))
        query["options"] = "-csearch_path=" + schema
        target = urlunsplit(parts._replace(query=urlencode(query)))
        asyncio.run(check(target))
    finally:
        with psycopg.connect(base, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


if __name__ == "__main__":
    main()
