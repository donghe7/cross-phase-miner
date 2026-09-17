"""Loss, retries, crash recovery, and transaction boundaries for arrival delivery."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from server import config
from server.outbox import ArrivalOutbox, DurableArrivalClient
from server.service import SignalService
from server.store import StateStore


def open_service(path):
    return SignalService(
        num_intersections=12,
        zone_size=8,
        num_hubs=2,
        speed=0,
        sim_start=config.default_sim_start() + 3600,
        db_path=str(path),
    )


def report(iid="seongsu_station", record_id="durable-1"):
    now = config.default_sim_start() + 3600
    return dict(
        robot_id="robot_001",
        intersection_id=iid,
        record_id=record_id,
        arrival_time=now - 200,
        depart_time=now,
        mode="scout",
        waited=200,
        transitions=[
            dict(
                timestamp=now,
                from_color="RED",
                to_color="GREEN",
                episode_start=now - 100,
                exact_red=True,
            )
        ],
    )


class DeliveryTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "state.sqlite3"

    def service(self):
        service = open_service(self.path)
        self.addCleanup(service.shutdown)
        return service

    def outbox(self, client, scope="test"):
        outbox = ArrivalOutbox(str(Path(self.folder.name) / "outbox.sqlite3"), scope, client, False)
        self.addCleanup(outbox.close)
        return outbox

    def test_process_exit_after_acceptance_resumes_learning_once(self):
        code = """
import os, sys
from unittest.mock import patch
from tests.server.test_delivery import open_service, report
with patch('server.service.SignalService._learn_worker'):
    service=open_service(sys.argv[1])
    ack=service.submit_arrival(**report())
    assert ack['stored']
    os._exit(0)
"""
        subprocess.run(
            [sys.executable, "-c", code, str(self.path)],
            check=True,
            cwd=Path(__file__).resolve().parents[2],
        )
        service = self.service()
        service.wait_for_learning()
        self.assertEqual(service.detail("seongsu_station")["learning"]["samples"], 1)
        self.assertTrue(service.submit_arrival(**report())["duplicate"])
        service.wait_for_learning()
        self.assertEqual(service.detail("seongsu_station")["learning"]["samples"], 1)

    def test_failed_commit_rolls_back_models_and_retries_without_double_learning(self):
        service = self.service()
        original = service.store._save_payload
        calls = []

        def fail_mid_transaction(cursor, table, key, data):
            original(cursor, table, key, data)
            if not calls:
                calls.append(1)
                raise RuntimeError("database interruption")

        with patch.object(service.store, "_save_payload", side_effect=fail_mid_transaction):
            service.submit_arrival(**report())
            service.wait_for_learning()
        self.assertEqual(service.stats.learn_errors, 1)
        self.assertEqual(service.detail("seongsu_station")["learning"]["samples"], 1)
        self.assertEqual(service.store.load_exact_red()[("seongsu_station", "day")], [100])
        self.assertEqual(service.store.load_learner(("seongsu_station", "day")).sample_count, 1)

    def test_lost_commit_response_restores_authoritative_state(self):
        service = self.service()
        original = service.store.save_learning_state
        calls = []

        def commit_then_disconnect(*args, **kwargs):
            result = original(*args, **kwargs)
            if not calls:
                calls.append(1)
                raise RuntimeError("commit response lost")
            return result

        with patch.object(service.store, "save_learning_state", side_effect=commit_then_disconnect):
            service.submit_arrival(**report())
            # wait for worker's recovery, not just the already committed job status
            import time

            deadline = time.monotonic() + 5
            while service.detail("seongsu_station")["learning"]["samples"] != 1:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
        service.submit_arrival(**report(record_id="second"))
        service.wait_for_learning()
        self.assertEqual(service.detail("seongsu_station")["learning"]["samples"], 1)

    def test_refresh_retries_after_commit_and_database_outage(self):
        service = self.service()
        commit = service.store.save_learning_state
        load = service.store.load_models
        committed = []
        failures = []

        def lose_response(*args, **kwargs):
            result = commit(*args, **kwargs)
            if not committed:
                committed.append(True)
                raise RuntimeError("response lost")
            return result

        def temporarily_unavailable():
            if len(failures) < 2:
                failures.append(True)
                raise RuntimeError("database offline")
            return load()

        import time

        with (
            patch.object(service.store, "save_learning_state", side_effect=lose_response),
            patch.object(service.store, "load_models", side_effect=temporarily_unavailable),
        ):
            service.submit_arrival(**report())
            deadline = time.monotonic() + 5
            while service.detail("seongsu_station")["learning"]["samples"] != 1:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
        self.assertEqual(len(failures), 2)
        self.assertEqual(service.store.pending_jobs(), {})

    def test_arrival_and_learning_job_insert_roll_back_together(self):
        store = StateStore(str(self.path))
        self.addCleanup(store.close)
        r = dict(report(), period="day", recorded_at=1, action="CROSS")
        with self.assertRaises(TypeError):
            store.save_visit(r, {"not_serializable": object()})
        self.assertEqual(store.load_visit_totals(), {})
        self.assertEqual(store.pending_jobs(), {})

    def test_outbox_survives_offline_restart_and_lost_ack(self):
        service = self.service()
        client = Mock()
        client.post.side_effect = OSError("offline")
        path = str(Path(self.folder.name) / "outbox.sqlite3")
        first = ArrivalOutbox(path, "test", client, False)
        first.enqueue(report())
        first.deliver_once()
        self.assertEqual(first.pending_count(), 1)
        first.close()

        def lost_ack(path, payload):
            service.submit_arrival(**payload)
            raise OSError("response lost")

        client.post.side_effect = lost_ack
        second = ArrivalOutbox(path, "test", client, False)
        second.deliver_once()
        self.assertEqual(second.pending_count(), 1)
        second.close()
        client.post.side_effect = lambda path, payload: service.submit_arrival(**payload)
        third = self.outbox(client)
        third.deliver_once()
        self.assertEqual(third.pending_count(), 0)
        service.wait_for_learning()
        self.assertEqual(service.detail("seongsu_station")["learning"]["samples"], 1)
        self.assertEqual(service.store.load_visit_totals()["seongsu_station"]["visits"], 1)
        self.assertEqual(
            {c.args[1]["record_id"] for c in client.post.call_args_list}, {"durable-1"}
        )

    def test_only_matching_application_ack_removes_a_report(self):
        client = Mock()
        client.post.return_value = {}  # MQTT publish accepted, but no application ack
        outbox = self.outbox(client)
        outbox.enqueue(report())
        outbox.deliver_once()
        for ack in (
            {"record_id": "durable-1", "robot_id": "robot_001"},
            {"stored": True, "record_id": "wrong", "robot_id": "robot_001"},
            {"stored": True, "record_id": "durable-1", "robot_id": "wrong"},
        ):
            outbox.acknowledge(ack)
            self.assertEqual(outbox.pending_count(), 1)
        outbox.acknowledge({"stored": True, "record_id": "durable-1", "robot_id": "robot_001"})
        self.assertEqual(outbox.pending_count(), 0)

    def test_outbox_scopes_do_not_mix_destinations(self):
        client = Mock()
        first = self.outbox(client, "first-server")
        second = self.outbox(client, "second-server")
        first.enqueue(report())
        second.deliver_once()
        client.post.assert_not_called()
        self.assertEqual(first.pending_count(), 1)

    def test_durable_adapter_journals_before_network_send(self):
        client = Mock()
        outbox = self.outbox(client)
        adapter = DurableArrivalClient(client, outbox)
        adapter.post("/v1/arrivals", report())
        client.post.assert_not_called()
        self.assertEqual(outbox.pending_count(), 1)
        adapter.post("/v1/travel", {"robot_id": "robot_001"})
        client.post.assert_called_once()

    def test_late_arrival_does_not_rewind_current_robot_location(self):
        service = self.service()
        now = service.clock.now()
        service.report_travel("robot_001", service.order[1], now + 1, now + 10)
        service.submit_arrival(**report())
        self.assertEqual(service.robots["robot_001"].action, "TRAVEL")
        self.assertEqual(service.robots["robot_001"].destination_id, service.order[1])
