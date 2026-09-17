"""Storage contract. PostgreSQL uses an isolated schema per test."""

import os
import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from crossphase_miner.core.learners import BayesianPeriodLearner
from crossphase_miner.core.models import PeriodModel, PhaseTransition, SignalColor
from server.migrate_sqlite import read_snapshot
from server.service import SignalService
from server.store import StateStore, learner_to_dict


class StoreContract:
    @staticmethod
    def visit_record(index=0):
        return dict(
            record_id=f"visit_{index:03}",
            intersection_id="source",
            robot_id=f"robot_{index % 2}",
            mode="scout",
            action="CROSS",
            recorded_at=1000 + index,
            arrival_time=900 + index,
            depart_time=1000 + index,
            waited=100,
            period="day",
            transitions=3,
            transitions_by_period={"day": 3},
        )

    def test_visit_and_job_are_atomic_and_checkpoint_completes_job_once(self):
        store = StateStore(self.target)
        self.addCleanup(store.close)
        record = self.visit_record()
        with self.assertRaises(TypeError):
            store.save_visit(record, {"bad": object()})
        self.assertEqual(store.load_visit_totals(), {})
        job = dict(robot_id=record["robot_id"], transitions=[], exacts=[])
        self.assertTrue(store.save_visit(record, job))
        self.assertFalse(store.save_visit(record, job))
        self.assertEqual(store.pending_jobs(), {"source": 1})
        self.assertEqual(store.next_learning_job()["record_id"], record["record_id"])
        key = ("source", "day")
        entries = [(key, PeriodModel(sample_count=3), BayesianPeriodLearner(), [100])]
        save = store._save_payload

        def fail_after_model(cursor, table, key, data):
            save(cursor, table, key, data)
            raise RuntimeError("interrupted transaction")

        with patch.object(store, "_save_payload", side_effect=fail_after_model):
            with self.assertRaises(RuntimeError):
                store.save_learning_state(entries, job_id=record["record_id"])
        self.assertEqual(store.load_models(), {})
        self.assertEqual(store.pending_jobs(), {"source": 1})
        self.assertTrue(store.save_learning_state(entries, job_id=record["record_id"]))
        self.assertFalse(
            store.save_learning_state(
                [(key, PeriodModel(sample_count=999), BayesianPeriodLearner(), [])],
                job_id=record["record_id"],
            )
        )
        self.assertEqual(store.load_models()[key].sample_count, 3)
        self.assertEqual(store.pending_jobs(), {})
        store.clear()
        self.assertEqual(store.pending_jobs(), {})
        self.assertTrue(store.save_visit(record, job))

    def test_migration_preserves_pending_learning_jobs(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.sqlite3"
            store = StateStore(str(source))
            store.upsert_intersections([("source", "Source", "District", 0, 0, 1)])
            store.save_visit(
                self.visit_record(), dict(robot_id="robot_0", transitions=[], exacts=[])
            )
            store.close()
            snapshot = read_snapshot(source)
            target = StateStore(self.target)
            self.addCleanup(target.close)
            target.import_snapshot(snapshot)
            self.assertEqual(target.pending_jobs(), {"source": 1})
            self.assertEqual(target.next_learning_job()["record_id"], "visit_000")

    def test_visit_history_window_deduplication_and_model_metadata(self):
        store = StateStore(self.target)
        history = {("source", "day"): {"first_model": {"samples": 3, "sim_time": 1000}}}
        try:
            for index in range(25):
                self.assertTrue(store.save_visit(self.visit_record(index)))
            self.assertFalse(store.save_visit(self.visit_record(0)))
            store.save_learning_state(
                [(("source", "day"), PeriodModel(sample_count=3), BayesianPeriodLearner(), [])],
                history,
            )
        finally:
            store.close()
        store = StateStore(self.target)
        try:
            self.assertEqual(store.load_model_history(), history)
            totals = store.load_visit_totals()["source"]
            self.assertEqual(totals["visits"], 25)
            self.assertEqual(len(totals["robot_ids"]), 2)
            self.assertEqual(
                [r["record_id"] for r in store.recent_visits("source")],
                [f"visit_{i:03}" for i in range(24, 4, -1)],
            )
            yields = store.load_recent_visit_yields()["source"]
            self.assertEqual(len(yields), 20)
            self.assertEqual(yields[0]["record_id"], "visit_005")
            self.assertEqual(yields[-1]["transitions_by_period"], {"day": 3})
            store.clear()
            self.assertEqual(store.load_visit_totals(), {})
            self.assertEqual(store.load_model_history(), {})
        finally:
            store.close()

    def test_sqlite_snapshot_import_is_atomic_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.sqlite3"
            store = StateStore(str(source))
            store.upsert_intersections([("source", "Source", "Seongsu", 0, 0, 1)])
            history = {("source", "day"): {"first_model": {"sim_time": 1000}}}
            store.save_learning_state(
                [(("source", "day"), PeriodModel(T_cycle=155), BayesianPeriodLearner(), [120])],
                history,
            )
            store.save_visit(self.visit_record())
            store.close()
            snapshot = read_snapshot(source)
            target = StateStore(self.target)
            try:
                target.import_snapshot(snapshot)
                self.assertEqual(target.load_models()[("source", "day")].T_cycle, 155)
                self.assertEqual(target.load_model_history(), history)
                self.assertEqual(target.recent_visits("source"), [self.visit_record()])
                with self.assertRaises(ValueError):
                    target.import_snapshot(snapshot)
                self.assertEqual(target.count_intersections(), 1)
                self.assertEqual(read_snapshot(source), snapshot)
            finally:
                target.close()

    def test_round_trip_upsert_and_reopen(self):
        key = ("성수's_station", "day")
        store = StateStore(self.target)
        store.upsert_intersections([(key[0], "圣水 · 성수", "Seongsu", 0, 0, 1)])
        store.upsert_intersections([(key[0], "Updated", "Seongsu", 2, 3, 1)])
        self.assertEqual(store.count_intersections(), 1)
        learner = BayesianPeriodLearner()
        learner.mu_T_cycle = 150
        model = PeriodModel(T_cycle=150, T_red=120, T_green=30, confidence=0.9, sample_count=12)
        store.save_learning_state([(key, model, learner, [120.0, 119.5])])
        other = StateStore(self.target)
        try:
            self.assertEqual(other.load_models()[key].T_cycle, 150)
            self.assertEqual(learner_to_dict(other.load_learner(key)), learner_to_dict(learner))
            self.assertEqual(other.load_exact_red()[key], [120.0, 119.5])
        finally:
            other.close()
            store.close()
        reopened = StateStore(self.target)
        try:
            self.assertEqual(reopened.load_models()[key].sample_count, 12)
            self.assertIsNone(reopened.load_learner(("missing", "day")))
        finally:
            reopened.close()

    def test_failed_checkpoint_rolls_back_and_connection_recovers(self):
        store = StateStore(self.target)
        key = ("test", "day")
        try:
            store.save_model(key, PeriodModel(T_cycle=120))
            with self.assertRaises(AttributeError):
                store.save_learning_state([(key, PeriodModel(T_cycle=150), None, [])])
            self.assertEqual(store.load_models()[key].T_cycle, 120)
            store.save_model(key, PeriodModel(T_cycle=140))
            self.assertEqual(store.load_models()[key].T_cycle, 140)
        finally:
            store.close()

    def test_explicit_reset_only_clears_application_tables(self):
        store = StateStore(self.target)
        try:
            store.upsert_intersections([("test", "Test", "Test", 0, 0, 1)])
            store.save_learning_state(
                [(("test", "day"), PeriodModel(), BayesianPeriodLearner(), [80])]
            )
            with store._transaction() as cursor:
                cursor.execute("CREATE TABLE unrelated_data (value INTEGER)")
                cursor.execute("INSERT INTO unrelated_data VALUES (42)")
            store.clear()
            self.assertEqual(store.count_intersections(), 0)
            self.assertEqual(store.load_models(), {})
            self.assertEqual(store.load_exact_red(), {})
            self.assertIsNone(store.load_learner(("test", "day")))
            with store._transaction() as cursor:
                cursor.execute("SELECT value FROM unrelated_data")
                self.assertEqual(cursor.fetchone()[0], 42)
        finally:
            store.close()

    def test_service_checkpoint_is_complete_before_shutdown(self):
        service = SignalService(
            num_intersections=12, zone_size=8, num_hubs=2, speed=0, db_path=self.target
        )
        try:
            iid = service.order[0]
            now = service.clock.now()
            transition = PhaseTransition(
                intersection_id=iid,
                timestamp=now,
                from_color=SignalColor.RED,
                to_color=SignalColor.GREEN,
                robot_id="robot_001",
                episode_start=now - 120,
            )
            service._learn(iid, [transition], [("day", 120.0)])
            observer = StateStore(self.target)
            try:
                self.assertIn((iid, "day"), observer.load_models())
                self.assertEqual(observer.load_learner((iid, "day")).sample_count, 1)
                self.assertEqual(observer.load_exact_red()[(iid, "day")], [120.0])
            finally:
                observer.close()
        finally:
            service.shutdown()
        restored = SignalService(
            num_intersections=12, zone_size=8, num_hubs=2, speed=0, db_path=self.target
        )
        try:
            self.assertEqual(list(restored._exact_red[(iid, "day")]), [120.0])
            self.assertEqual(restored.learners.get((iid, "day")).sample_count, 1)
        finally:
            restored.shutdown()


class SQLiteStoreTest(StoreContract, unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.target = str(Path(self.directory.name) / "state.sqlite3")

    def tearDown(self):
        self.directory.cleanup()

    def test_upgrade_and_import_of_database_without_history_tables(self):
        store = StateStore(self.target)
        store.save_model(("old", "day"), PeriodModel(sample_count=12, confidence=0.8))
        store.close()
        with sqlite3.connect(self.target) as connection:
            for table in ("red_measurements", "model_history", "visits"):
                connection.execute(f"DROP TABLE {table}")
        snapshot = read_snapshot(Path(self.target))
        self.assertEqual(snapshot["model_history"], [])
        self.assertEqual(snapshot["visits"], [])
        upgraded = StateStore(self.target)
        try:
            self.assertEqual(upgraded.load_models()[("old", "day")].sample_count, 12)
            self.assertEqual(upgraded.load_model_history(), {})
            self.assertEqual(upgraded.load_visit_totals(), {})
            self.assertTrue(upgraded.save_visit(self.visit_record()))
        finally:
            upgraded.close()


@unittest.skipUnless(os.environ.get("CP_TEST_POSTGRES_URL"), "CP_TEST_POSTGRES_URL not set")
class PostgresStoreTest(StoreContract, unittest.TestCase):
    def setUp(self):
        import psycopg
        from psycopg import sql

        self.base = os.environ["CP_TEST_POSTGRES_URL"]
        self.schema = "cp_test_" + uuid.uuid4().hex
        with psycopg.connect(self.base, autocommit=True) as connection:
            connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.schema)))
        parts = urlsplit(self.base)
        query = dict(parse_qsl(parts.query))
        query["options"] = "-csearch_path=" + self.schema
        self.target = urlunsplit(parts._replace(query=urlencode(query)))

    def test_broken_connection_reconnects_without_losing_pending_job(self):
        store = StateStore(self.target)
        self.addCleanup(store.close)
        store.save_visit(self.visit_record(), dict(robot_id="robot_0", transitions=[], exacts=[]))
        store._conn.close()  # simulate a socket closed by a restarted database
        self.assertEqual(store.pending_jobs(), {"source": 1})
        self.assertTrue(store.save_learning_state([], job_id="visit_000"))
        self.assertEqual(store.pending_jobs(), {})

    def tearDown(self):
        import psycopg
        from psycopg import sql

        with psycopg.connect(self.base, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(self.schema))
            )


if __name__ == "__main__":
    unittest.main()
