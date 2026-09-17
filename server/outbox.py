"""Durable arrival delivery. Broker receipt alone never removes a local report."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

LOG = logging.getLogger(__name__)


class ArrivalOutbox:
    """One fleet-local SQLite journal, scoped to a logical server/broker destination."""

    def __init__(self, path: str, scope: str, client, autostart: bool = True):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.scope = scope
        self.client = client
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._closed = False
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        with self._db:
            self._db.execute("""CREATE TABLE IF NOT EXISTS arrival_outbox (
                scope TEXT NOT NULL, record_id TEXT NOT NULL, robot_id TEXT NOT NULL,
                payload TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt REAL NOT NULL DEFAULT 0, created_at REAL NOT NULL,
                PRIMARY KEY (scope, record_id))""")
            # Previously sent but unconfirmed reports are immediately eligible on restart.
            self._db.execute("UPDATE arrival_outbox SET next_attempt=0 WHERE scope=?", (scope,))
        self._thread = threading.Thread(target=self._run, name="arrival-outbox", daemon=True)
        if autostart:
            self._thread.start()

    def enqueue(self, report: dict) -> None:
        encoded = json.dumps(report, allow_nan=False)
        with self._lock, self._db:
            if self._closed:
                raise RuntimeError("Outbox closed")
            self._db.execute(
                "INSERT INTO arrival_outbox "
                "(scope, record_id, robot_id, payload, created_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(scope, record_id) DO NOTHING",
                (self.scope, report["record_id"], report["robot_id"], encoded, time.time()),
            )
        self._wake.set()

    def acknowledge(self, ack: dict) -> None:
        if ack.get("stored") is not True or not ack.get("record_id") or not ack.get("robot_id"):
            return
        with self._lock:
            if self._closed:
                return
            with self._db:
                self._db.execute(
                    "DELETE FROM arrival_outbox WHERE scope=? AND record_id=? AND robot_id=?",
                    (self.scope, ack["record_id"], ack["robot_id"]),
                )

    def pending_count(self) -> int:
        with self._lock:
            return self._db.execute(
                "SELECT COUNT(*) FROM arrival_outbox WHERE scope=?", (self.scope,)
            ).fetchone()[0]

    def deliver_once(self) -> None:
        with self._lock:
            rows = self._db.execute(
                "SELECT record_id, payload, attempts FROM arrival_outbox "
                "WHERE scope=? AND next_attempt<=? ORDER BY created_at, record_id LIMIT 64",
                (self.scope, time.time()),
            ).fetchall()
        for record_id, encoded, attempts in rows:
            if self._stop.is_set():
                return
            try:
                ack = self.client.post("/v1/arrivals", json.loads(encoded))
                # HTTP responds synchronously; MQTT provides a separate application ack.
                if isinstance(ack, dict) and ack.get("record_id") == record_id:
                    self.acknowledge(ack)
            except Exception as error:
                LOG.warning(
                    "Arrival %s unconfirmed; retained for retry (%s)",
                    record_id,
                    type(error).__name__,
                )
            with self._lock, self._db:
                self._db.execute(
                    "UPDATE arrival_outbox SET attempts=attempts+1, next_attempt=? "
                    "WHERE scope=? AND record_id=?",
                    (time.time() + min(30.0, 0.5 * 2 ** min(attempts, 6)), self.scope, record_id),
                )

    def _run(self):
        while not self._stop.is_set():
            try:
                self.deliver_once()
            except Exception:
                LOG.exception("Outbox delivery failed; persisted reports retained")
            self._wake.wait(0.25)
            self._wake.clear()

    def close(self):
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join()
        with self._lock:
            self._closed = True
            self._db.close()


class DurableArrivalClient:
    """Keep live telemetry best effort, but journal completed arrivals before sending."""

    def __init__(self, client, outbox: ArrivalOutbox):
        self.client = client
        self.outbox = outbox

    def get(self, path):
        return self.client.get(path)

    def post(self, path, payload):
        if path == "/v1/arrivals":
            self.outbox.enqueue(payload)
            return {"locally_queued": True}
        return self.client.post(path, payload)
