"""
Persistence and the learner working set.

Why this exists: a Bayesian learner is cheap (a few hundred bytes) but a
city has more crossings than a single process should keep warm forever,
and a restart must not throw away a day of learning.  So:

* every learned model and learner state can be serialised to the database;
* only ``LEARNER_CACHE_SIZE`` learners stay in memory (LRU).  Evicting one
  writes it out; touching a cold crossing reads it back.

Serialisation lives here rather than on the learner classes so the
``crossphase_miner`` package stays free of storage concerns.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import OrderedDict, deque
from contextlib import contextmanager
from typing import Dict, Iterable, Optional, Tuple

from crossphase_miner.core.learners import BayesianPeriodLearner
from crossphase_miner.core.models import PeriodModel

Key = Tuple[str, str]  # (intersection_id, tod_period)


# --- serialisation --------------------------------------------------------


def learner_to_dict(learner: BayesianPeriodLearner) -> dict:
    """Serialise a Bayesian learner's full state."""
    return {
        "prior_T_cycle": list(learner.prior_T_cycle),
        "mu_T_cycle": learner.mu_T_cycle,
        "sigma_T_cycle": learner.sigma_T_cycle,
        "mu_T_red": learner.mu_T_red,
        "sigma_T_red": learner.sigma_T_red,
        "phi_offset": learner.phi_offset,
        "rg_times": list(learner._rg_times),
        "red_samples": list(learner._red_samples),
        "shared_T_red": learner._shared_T_red,
        "sample_count": learner.sample_count,
    }


def learner_from_dict(data: dict) -> BayesianPeriodLearner:
    """Restore a Bayesian learner from :func:`learner_to_dict` output."""
    learner = BayesianPeriodLearner(tuple(data["prior_T_cycle"]))
    learner.mu_T_cycle = data["mu_T_cycle"]
    learner.sigma_T_cycle = data["sigma_T_cycle"]
    learner.mu_T_red = data["mu_T_red"]
    learner.sigma_T_red = data["sigma_T_red"]
    learner.phi_offset = data["phi_offset"]
    learner._rg_times = deque(data["rg_times"], maxlen=50)
    learner._red_samples = deque(data["red_samples"], maxlen=50)
    learner._shared_T_red = data["shared_T_red"]
    learner.sample_count = data["sample_count"]
    return learner


def model_to_dict(model: PeriodModel) -> dict:
    return {
        "T_cycle": model.T_cycle,
        "T_red": model.T_red,
        "T_green": model.T_green,
        "phi_offset": model.phi_offset,
        "confidence": model.confidence,
        "sample_count": model.sample_count,
        "last_updated": model.last_updated,
    }


def model_from_dict(data: dict) -> PeriodModel:
    return PeriodModel(**data)


# --- durable storage ------------------------------------------------------


def is_postgres(target: str) -> bool:
    return target.startswith(("postgresql://", "postgres://"))


class StateStore:
    """One transactional interface for SQLite files and PostgreSQL URLs.

    A connection is shared by the service and its learner thread, protected by
    a lock. PostgreSQL uses explicit transactions on an autocommit connection
    so read operations never leave an idle transaction open.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.backend = "postgresql" if is_postgres(path) else "sqlite"
        self._lock = threading.Lock()
        if self.backend == "postgresql":
            try:
                import psycopg
                from psycopg.types.json import Jsonb
            except ImportError as error:
                raise RuntimeError(
                    "PostgreSQL requires psycopg: pip install -r server/requirements.txt"
                ) from error
            self._json = Jsonb
            self._conn = psycopg.connect(path, autocommit=True, connect_timeout=10)
        else:
            if "://" in path:
                raise ValueError("Use a SQLite file path or a postgresql:// URL")
            self._json = json.dumps
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=FULL")
        try:
            self._init_schema()
        except Exception:
            self._conn.close()
            raise

    def _sql(self, query: str) -> str:
        return query.replace("?", "%s") if self.backend == "postgresql" else query

    @contextmanager
    def _transaction(self):
        with self._lock:
            transaction = self._conn.transaction() if self.backend == "postgresql" else self._conn
            with transaction:
                cursor = self._conn.cursor()
                try:
                    yield cursor
                finally:
                    cursor.close()

    @staticmethod
    def _decode(payload):
        return json.loads(payload) if isinstance(payload, str) else payload

    def _init_schema(self) -> None:
        payload_type = "JSONB" if self.backend == "postgresql" else "TEXT"
        with self._transaction() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS intersections (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, district TEXT,
                    grid_x INTEGER, grid_y INTEGER, in_zone INTEGER DEFAULT 0
                )
            """)
            for table in ("models", "learners", "red_measurements", "model_history"):
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        id TEXT NOT NULL, period TEXT NOT NULL,
                        payload {payload_type} NOT NULL,
                        PRIMARY KEY (id, period)
                    )
                """)
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS visits (
                    record_id TEXT PRIMARY KEY, id TEXT NOT NULL,
                    robot_id TEXT NOT NULL, mode TEXT NOT NULL, action TEXT NOT NULL,
                    recorded_at DOUBLE PRECISION NOT NULL, payload {payload_type} NOT NULL
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS visits_crossing_time ON visits (id, recorded_at)"
            )

    def clear(self) -> None:
        """Explicit reset of CrossPhase tables only; never drop the database."""
        with self._transaction() as cursor:
            for table in (
                "visits",
                "model_history",
                "models",
                "learners",
                "red_measurements",
                "intersections",
            ):
                cursor.execute(f"DELETE FROM {table}")

    def upsert_intersections(self, rows: Iterable[tuple]) -> None:
        with self._transaction() as cursor:
            cursor.executemany(
                self._sql(
                    "INSERT INTO intersections (id, name, district, grid_x, grid_y,"
                    " in_zone) VALUES (?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(id) DO UPDATE SET name=excluded.name,"
                    " district=excluded.district, grid_x=excluded.grid_x,"
                    " grid_y=excluded.grid_y, in_zone=excluded.in_zone"
                ),
                rows,
            )

    def count_intersections(self) -> int:
        with self._transaction() as cursor:
            cursor.execute("SELECT COUNT(*) FROM intersections")
            return int(cursor.fetchone()[0])

    def _save_payload(self, cursor, table: str, key: Key, data) -> None:
        if table not in ("models", "learners", "red_measurements", "model_history"):
            raise ValueError("unknown state table")
        cursor.execute(
            self._sql(
                f"INSERT INTO {table} (id, period, payload) VALUES (?, ?, ?)"
                " ON CONFLICT(id, period) DO UPDATE SET payload=excluded.payload"
            ),
            (key[0], key[1], self._json(data)),
        )

    def save_model(self, key: Key, model: PeriodModel) -> None:
        with self._transaction() as cursor:
            self._save_payload(cursor, "models", key, model_to_dict(model))

    def load_models(self) -> Dict[Key, PeriodModel]:
        with self._transaction() as cursor:
            cursor.execute("SELECT id, period, payload FROM models")
            return {
                (iid, period): model_from_dict(self._decode(payload))
                for iid, period, payload in cursor.fetchall()
            }

    def save_learner(self, key: Key, learner: BayesianPeriodLearner) -> None:
        with self._transaction() as cursor:
            self._save_payload(cursor, "learners", key, learner_to_dict(learner))

    def load_learner(self, key: Key) -> Optional[BayesianPeriodLearner]:
        with self._transaction() as cursor:
            cursor.execute(
                self._sql("SELECT payload FROM learners WHERE id = ? AND period = ?"), key
            )
            row = cursor.fetchone()
            return learner_from_dict(self._decode(row[0])) if row else None

    def save_learning_state(self, entries: Iterable[tuple], history: Optional[dict] = None) -> None:
        """Commit all affected periods' models, learners and exact reds together."""
        with self._transaction() as cursor:
            for key, model, learner, exact_red in entries:
                self._save_payload(cursor, "models", key, model_to_dict(model))
                self._save_payload(cursor, "learners", key, learner_to_dict(learner))
                self._save_payload(cursor, "red_measurements", key, list(exact_red))
                if history and key in history:
                    self._save_payload(cursor, "model_history", key, history[key])

    def load_model_history(self) -> dict:
        with self._transaction() as cursor:
            cursor.execute("SELECT id, period, payload FROM model_history")
            return {
                (iid, period): self._decode(payload) for iid, period, payload in cursor.fetchall()
            }

    def save_visit(self, record: dict) -> bool:
        with self._transaction() as cursor:
            cursor.execute(
                self._sql(
                    "INSERT INTO visits (record_id, id, robot_id, mode, action, recorded_at, payload)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(record_id) DO NOTHING"
                ),
                (
                    record["record_id"],
                    record["intersection_id"],
                    record["robot_id"],
                    record["mode"],
                    record["action"],
                    record["recorded_at"],
                    self._json(record),
                ),
            )
            return cursor.rowcount == 1

    def load_visit_totals(self) -> dict:
        totals = {}
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT id, robot_id, mode, action, COUNT(*) FROM visits GROUP BY id, robot_id, mode, action"
            )
            for iid, robot_id, mode, action, count in cursor.fetchall():
                total = totals.setdefault(
                    iid,
                    {
                        "robot_ids": set(),
                        "visits": 0,
                        "scout": 0,
                        "normal": 0,
                        "cross": 0,
                        "timeout": 0,
                    },
                )
                total["robot_ids"].add(robot_id)
                total["visits"] += count
                if mode in ("scout", "normal"):
                    total[mode] += count
                if action in ("CROSS", "TIMEOUT"):
                    total[action.lower()] += count
        return totals

    def recent_visits(self, intersection_id: str, limit: int = 20) -> list:
        with self._transaction() as cursor:
            cursor.execute(
                self._sql(
                    "SELECT payload FROM visits WHERE id = ? ORDER BY recorded_at DESC, record_id DESC LIMIT ?"
                ),
                (intersection_id, limit),
            )
            return [self._decode(row[0]) for row in cursor.fetchall()]

    def load_recent_visit_yields(self) -> dict:
        recent = {}
        with self._transaction() as cursor:
            cursor.execute("""
                SELECT id, payload FROM (
                    SELECT id, payload, recorded_at, record_id,
                           ROW_NUMBER() OVER (PARTITION BY id ORDER BY recorded_at DESC, record_id DESC) AS n
                    FROM visits
                ) ranked WHERE n <= 20 ORDER BY recorded_at, record_id
            """)
            for iid, payload in cursor.fetchall():
                recent.setdefault(iid, deque(maxlen=20)).append(self._decode(payload))
        return recent

    def load_exact_red(self) -> Dict[Key, list]:
        with self._transaction() as cursor:
            cursor.execute("SELECT id, period, payload FROM red_measurements")
            return {
                (iid, period): self._decode(payload) for iid, period, payload in cursor.fetchall()
            }

    def import_snapshot(self, snapshot: dict) -> None:
        """Import into an empty target in one transaction; never overwrite."""
        with self._transaction() as cursor:
            tables = (
                "intersections",
                "models",
                "learners",
                "red_measurements",
                "model_history",
                "visits",
            )
            if self.backend == "postgresql":
                cursor.execute(
                    "LOCK TABLE intersections, models, learners, red_measurements, model_history, visits IN EXCLUSIVE MODE"
                )
            else:
                cursor.execute("BEGIN IMMEDIATE")
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                if cursor.fetchone()[0]:
                    raise ValueError("Destination contains CrossPhase data; use an empty database")
            cursor.executemany(
                self._sql(
                    "INSERT INTO intersections (id, name, district, grid_x, grid_y, in_zone)"
                    " VALUES (?, ?, ?, ?, ?, ?)"
                ),
                snapshot["intersections"],
            )
            for table in tables[1:-1]:
                for iid, period, payload in snapshot.get(table, []):
                    self._save_payload(cursor, table, (iid, period), payload)
            for record in snapshot.get("visits", []):
                cursor.execute(
                    self._sql(
                        "INSERT INTO visits (record_id, id, robot_id, mode, action, recorded_at, payload)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?)"
                    ),
                    (
                        record["record_id"],
                        record["intersection_id"],
                        record["robot_id"],
                        record["mode"],
                        record["action"],
                        record["recorded_at"],
                        self._json(record),
                    ),
                )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# --- working set ----------------------------------------------------------


class LearnerCache:
    """
    LRU working set of Bayesian learners keyed by (crossing, TOD period).

    Misses are served from the database; evictions are written back.  ``hits``,
    ``misses`` and ``evictions`` are reported by ``GET /v1/stats`` so the
    cache can be sized against a real workload.
    """

    def __init__(self, store: StateStore, capacity: int) -> None:
        self.store = store
        self.capacity = max(1, capacity)
        self._items: "OrderedDict[Key, BayesianPeriodLearner]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.restores = 0

    def __len__(self) -> int:
        return len(self._items)

    def get(self, key: Key) -> BayesianPeriodLearner:
        """Return the learner for ``key``, creating or restoring as needed."""
        learner = self._items.get(key)
        if learner is not None:
            self.hits += 1
            self._items.move_to_end(key)
            return learner

        self.misses += 1
        learner = self.store.load_learner(key)
        if learner is not None:
            self.restores += 1
        else:
            learner = BayesianPeriodLearner()

        self._items[key] = learner
        self._items.move_to_end(key)
        self._evict_if_needed()
        return learner

    def peek(self, key: Key) -> Optional[BayesianPeriodLearner]:
        """Return an in-memory learner without touching storage or LRU order."""
        return self._items.get(key)

    def keys_for(self, intersection_id: str) -> list:
        """In-memory keys belonging to one crossing."""
        return [k for k in self._items if k[0] == intersection_id]

    def _evict_if_needed(self) -> None:
        while len(self._items) > self.capacity:
            key, learner = self._items.popitem(last=False)
            self.store.save_learner(key, learner)
            self.evictions += 1

    def flush(self) -> None:
        """Persist every in-memory learner."""
        for key, learner in self._items.items():
            self.store.save_learner(key, learner)
