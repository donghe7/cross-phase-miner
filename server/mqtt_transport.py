"""Local MQTT transport; database/query semantics remain in SignalService."""

import asyncio
import contextlib
import json
import logging
import os
import queue
import socket
import threading
import time
import uuid

import paho.mqtt.client as mqtt

from server import config

LOG = logging.getLogger(__name__)


def client(name):
    connection = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2, client_id=name + "-" + uuid.uuid4().hex[:12]
    )
    if os.environ.get("CP_MQTT_USER"):
        connection.username_pw_set(os.environ["CP_MQTT_USER"], os.environ.get("CP_MQTT_PASSWORD"))
    connection.reconnect_delay_set(1, 5)
    connection.on_socket_open = lambda client, userdata, sock: sock.setsockopt(
        socket.IPPROTO_TCP, socket.TCP_NODELAY, 1
    )
    connection.max_queued_messages_set(2000)
    return connection


class ServerMQTT:
    def __init__(self, service):
        self.service = service
        self.prefix = config.MQTT_PREFIX
        self.client = client("crossphase-server")
        self.connected = threading.Event()
        self.incoming = queue.Queue(maxsize=4096)
        self.errors = 0
        self.received = 0
        self.max_dispatch_ms = 0.0
        self.max_loop_gap_ms = 0.0
        self.sequences = {}
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = lambda *args: self.connected.clear()
        self.client.on_message = self.on_message
        self.task = None

    def on_connect(self, connection, userdata, flags, reason_code, properties):
        if not reason_code.is_failure:
            connection.subscribe([(self.prefix + "/robots/+/+", 1), (self.prefix + "/query/+", 0)])
            self.connected.set()

    def on_message(self, connection, userdata, message):
        if message.retain:
            return  # live telemetry must never be replayed from retained state
        try:
            self.incoming.put_nowait((message.topic, bytes(message.payload)))
        except queue.Full:
            self.errors += 1
            LOG.error("MQTT ingestion queue full")

    async def start(self):
        self.client.connect_async(config.MQTT_HOST, config.MQTT_PORT, keepalive=15)
        self.client.loop_start()
        for _ in range(100):
            if self.connected.is_set():
                self.task = asyncio.create_task(self.run())
                return
            await asyncio.sleep(0.05)
        self.client.disconnect()
        self.client.loop_stop()
        raise RuntimeError("MQTT broker unavailable; start the local broker first")

    def dispatch(self, topic, encoded):
        from server.app import ArrivalReport, ObservationBatch, TravelReport

        if topic.startswith(self.prefix + "/query/"):
            requester = topic.removeprefix(self.prefix + "/query/")
            request = json.loads(encoded)
            try:
                response = {
                    "request_id": request["request_id"],
                    "detail": self.service.detail(request["intersection_id"]),
                }
            except KeyError:
                response = {
                    "request_id": request.get("request_id"),
                    "error": "unknown crossing or missing field",
                }
            self.client.publish(self.prefix + "/replies/" + requester, json.dumps(response))
            return
        robot_id, kind = topic.removeprefix(self.prefix + "/robots/").split("/")
        envelope = json.loads(encoded)
        payload = envelope["payload"]
        if payload.get("robot_id") != robot_id:
            raise ValueError("robot ID does not match topic")
        session = envelope["session"]
        if not isinstance(session, str) or not session or any(c in session for c in "/+#"):
            raise ValueError("invalid reply session")
        key = (session, robot_id)
        seq = int(envelope["seq"])
        if kind != "arrivals" and seq <= self.sequences.get(key, -1):
            return
        if kind == "observations":
            batch = ObservationBatch.model_validate(payload)
            self.service.ingest_batch(
                batch.robot_id,
                batch.intersection_id,
                [(o.t, o.color, o.conf) for o in batch.observations],
                batch.mode,
                batch.action,
            )
        elif kind == "travel":
            report = TravelReport.model_validate(payload)
            if report.arrives_at < report.started_at:
                raise ValueError("arrival precedes departure")
            self.service.report_travel(**report.model_dump())
        elif kind == "arrivals":
            report = ArrivalReport.model_validate(payload)
            ack = self.service.submit_arrival(**report.model_dump())
            # Even an old sequence may be a retransmission after a lost application ack.
            self.client.publish(
                self.prefix + "/acks/" + session + "/" + robot_id,
                json.dumps(ack),
                qos=1,
                retain=False,
            )
        else:
            raise ValueError("unknown message kind")
        self.sequences[key] = max(seq, self.sequences.get(key, -1))
        self.received += 1

    async def run(self):
        next_state = 0.0
        previous_loop = time.monotonic()
        while True:
            loop_now = time.monotonic()
            self.max_loop_gap_ms = max(self.max_loop_gap_ms, (loop_now - previous_loop) * 1000)
            previous_loop = loop_now
            for _ in range(200):
                try:
                    topic, payload = self.incoming.get_nowait()
                except queue.Empty:
                    break
                try:
                    started = time.monotonic()
                    self.dispatch(topic, payload)
                except Exception as error:
                    self.errors += 1
                    LOG.warning(
                        "MQTT dispatch failed (%s); no arrival ack sent", type(error).__name__
                    )
                finally:
                    self.max_dispatch_ms = max(
                        self.max_dispatch_ms, (time.monotonic() - started) * 1000
                    )
            if self.connected.is_set():
                self.client.publish(
                    self.prefix + "/clock", json.dumps(self.service.clock.as_dict())
                )
                if time.monotonic() >= next_state:
                    now = self.service.clock.now()
                    for iid in list(self.service.live):
                        self.client.publish(
                            self.prefix + "/crossings/" + iid + "/state",
                            json.dumps(self.service.predict(iid, now)),
                        )
                    next_state = time.monotonic() + 0.2
            await asyncio.sleep(min(0.05, 0.25 / max(1, self.service.clock.speed)))

    async def close(self):
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
        self.client.disconnect()
        self.client.loop_stop()

    def status(self):
        return {
            "enabled": True,
            "connected": self.connected.is_set(),
            "received": self.received,
            "errors": self.errors,
            "queue_depth": self.incoming.qsize(),
            "max_dispatch_ms": round(self.max_dispatch_ms, 2),
            "max_loop_gap_ms": round(self.max_loop_gap_ms, 2),
        }


class FleetMQTT:
    def __init__(self, clock):
        self.clock = clock
        self.client = client("crossphase-fleet")
        self.connected = threading.Event()
        self.session = uuid.uuid4().hex
        self.arrival_ack_handler = None
        self.lock = threading.Lock()
        self.seq = 0
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = lambda *args: self.connected.clear()
        self.client.on_message = self.on_message
        self.client.connect_async(config.MQTT_HOST, config.MQTT_PORT, keepalive=15)
        self.client.loop_start()
        if not self.connected.wait(5):
            self.close()
            raise OSError("MQTT broker unavailable")

    def on_connect(self, connection, userdata, flags, reason_code, properties):
        if not reason_code.is_failure:
            connection.subscribe(
                [
                    (config.MQTT_PREFIX + "/clock", 0),
                    (config.MQTT_PREFIX + "/acks/" + self.session + "/+", 1),
                ]
            )
            self.connected.set()

    def on_message(self, connection, userdata, message):
        if not message.retain:
            try:
                payload = json.loads(message.payload)
                if message.topic == config.MQTT_PREFIX + "/clock":
                    self.clock.synchronize(payload)
                elif (
                    message.topic.startswith(config.MQTT_PREFIX + "/acks/" + self.session + "/")
                    and message.topic.rsplit("/", 1)[-1] == payload.get("robot_id")
                    and self.arrival_ack_handler
                ):
                    self.arrival_ack_handler(payload)
            except Exception:
                # A local journal error must not kill the MQTT network loop. The
                # report stays pending and its application ack will be requested again.
                LOG.exception("MQTT clock/ack handling failed; unconfirmed reports retained")

    def post(self, path, payload):
        if not self.connected.is_set():
            raise OSError("MQTT disconnected")
        kind = path.removeprefix("/v1/")
        with self.lock:
            self.seq += 1
            result = self.client.publish(
                config.MQTT_PREFIX + "/robots/" + payload["robot_id"] + "/" + kind,
                json.dumps({"session": self.session, "seq": self.seq, "payload": payload}),
                qos=0 if kind == "observations" else 1,
                retain=False,
            )
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            raise OSError("MQTT publish not queued")
        return {}

    def close(self):
        self.client.disconnect()
        self.client.loop_stop()
