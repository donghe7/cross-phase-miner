"""MQTT envelopes preserve per-robot order and durable arrival deduplication."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from server import config
from server.mqtt_transport import ServerMQTT
from server.service import SignalService


class MQTTTransportTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.service = SignalService(
            num_intersections=12,
            zone_size=8,
            num_hubs=2,
            speed=0,
            sim_start=config.default_sim_start() + 3600,
            db_path=str(Path(self.folder.name) / "state.sqlite3"),
        )
        self.bridge = ServerMQTT(self.service)
        self.iid = self.service.order[0]
        self.now = self.service.clock.now()

    def tearDown(self):
        self.service.shutdown()
        self.folder.cleanup()

    def send(self, kind, seq, **payload):
        payload["robot_id"] = "robot_001"
        self.bridge.dispatch(
            self.bridge.prefix + "/robots/robot_001/" + kind,
            json.dumps(dict(session="test", seq=seq, payload=payload)),
        )

    def test_older_and_duplicate_observations_do_not_rewind_a_trip(self):
        frame = dict(
            intersection_id=self.iid, observations=[dict(t=self.now, color="RED", conf=0.95)]
        )
        self.send("observations", 1, **frame)
        self.send("observations", 1, **frame)
        self.assertEqual(self.service.stats.observations, 1)
        self.send(
            "travel",
            3,
            destination_id=self.service.order[1],
            started_at=self.now + 1,
            arrives_at=self.now + 30,
        )
        self.send("observations", 2, **frame)
        self.assertEqual(self.service.robots["robot_001"].action, "TRAVEL")
        self.assertIsNone(self.service.robots["robot_001"].intersection_id)

    def test_arrival_duplicate_is_not_learned_or_counted_twice(self):
        report = dict(
            intersection_id=self.iid,
            record_id="arrival-1",
            arrival_time=self.now - 180,
            depart_time=self.now,
            mode="scout",
            waited=180,
            transitions=[dict(timestamp=self.now, from_color="RED", to_color="GREEN")],
        )
        self.send("arrivals", 1, **report)
        self.send("arrivals", 1, **report)
        self.send("arrivals", 2, **report)
        self.service.wait_for_learning()
        detail = self.service.detail(self.iid)
        self.assertEqual(detail["learning"]["visits"]["visits"], 1)
        self.assertEqual(detail["learning"]["samples"], 1)

    def test_retransmitted_arrival_is_acknowledged_even_after_a_newer_sequence(self):
        self.bridge.client.publish = Mock()
        report = dict(
            intersection_id=self.iid,
            record_id="ack-1",
            arrival_time=self.now - 10,
            depart_time=self.now,
            mode="normal",
            waited=10,
            transitions=[],
        )
        self.send(
            "travel",
            10,
            destination_id=self.service.order[1],
            started_at=self.now + 1,
            arrives_at=self.now + 20,
        )
        self.send("arrivals", 1, **report)
        self.send("arrivals", 1, **report)
        self.assertEqual(self.bridge.client.publish.call_count, 2)
        topic, payload = self.bridge.client.publish.call_args.args
        ack = json.loads(payload)
        self.assertEqual(topic, self.bridge.prefix + "/acks/test/robot_001")
        self.assertTrue(ack["stored"])
        self.assertTrue(ack["duplicate"])
        self.assertEqual(ack["record_id"], "ack-1")
        self.assertEqual(self.bridge.sequences[("test", "robot_001")], 10)
        self.assertEqual(self.service.robots["robot_001"].action, "TRAVEL")

    def test_database_failure_never_emits_a_stored_ack(self):
        self.bridge.client.publish = Mock()
        with patch.object(
            self.service.store, "save_visit", side_effect=RuntimeError("DB unavailable")
        ):
            with self.assertRaises(RuntimeError):
                self.send(
                    "arrivals",
                    1,
                    intersection_id=self.iid,
                    record_id="not-stored",
                    arrival_time=self.now,
                    depart_time=self.now,
                    waited=0,
                    transitions=[],
                )
        self.bridge.client.publish.assert_not_called()
        self.assertEqual(self.service.store.load_visit_totals(), {})

    def test_query_reply_has_correlation_and_both_observation_and_prediction(self):
        self.bridge.client.publish = Mock()
        self.bridge.dispatch(
            self.bridge.prefix + "/query/inspector",
            json.dumps(dict(request_id="query-1", intersection_id=self.iid)),
        )
        topic, encoded = self.bridge.client.publish.call_args.args
        self.assertEqual(topic, self.bridge.prefix + "/replies/inspector")
        reply = json.loads(encoded)
        self.assertEqual(reply["request_id"], "query-1")
        self.assertIn("live", reply["detail"])
        self.assertIn("model_prediction", reply["detail"])

    def test_topic_identity_must_match_payload(self):
        with self.assertRaises(ValueError):
            self.bridge.dispatch(
                self.bridge.prefix + "/robots/wrong/observations",
                json.dumps(dict(session="test", seq=1, payload=dict(robot_id="robot_001"))),
            )
        self.assertEqual(self.service.stats.observations, 0)


if __name__ == "__main__":
    unittest.main()
