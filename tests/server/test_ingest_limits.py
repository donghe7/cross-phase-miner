"""A single misbehaving client must not grow the server unbounded.

There is no robot/API authentication yet (see docs/architecture.md), so the
request models are the only backstop against a client that sends an empty
or absurdly long ``robot_id``/``intersection_id`` (unbounded keys in
``SignalService.robots`` / ``LiveSignalState.votes``) or an oversized batch
(unbounded per-request allocation on the event loop).
"""

import unittest

from pydantic import ValidationError

from server import config
from server.app import ArrivalReport, ObservationBatch, TravelReport


class ObservationBatchLimitsTest(unittest.TestCase):
    def test_normal_batch_is_accepted(self):
        batch = ObservationBatch(
            robot_id="robot_001",
            intersection_id="seongsu_station",
            observations=[dict(t=0.0, color="RED", conf=0.9)],
        )
        self.assertEqual(batch.robot_id, "robot_001")

    def test_empty_robot_id_is_rejected(self):
        with self.assertRaises(ValidationError):
            ObservationBatch(robot_id="", intersection_id="a", observations=[])

    def test_overlong_robot_id_is_rejected(self):
        with self.assertRaises(ValidationError):
            ObservationBatch(
                robot_id="x" * (config.MAX_ID_LENGTH + 1),
                intersection_id="a",
                observations=[],
            )

    def test_overlong_intersection_id_is_rejected(self):
        with self.assertRaises(ValidationError):
            ObservationBatch(
                robot_id="robot_001",
                intersection_id="x" * (config.MAX_ID_LENGTH + 1),
                observations=[],
            )

    def test_oversized_batch_is_rejected(self):
        frame = dict(t=0.0, color="RED", conf=0.9)
        with self.assertRaises(ValidationError):
            ObservationBatch(
                robot_id="robot_001",
                intersection_id="a",
                observations=[frame] * (config.MAX_OBSERVATIONS_PER_BATCH + 1),
            )

    def test_batch_at_the_limit_is_accepted(self):
        frame = dict(t=0.0, color="RED", conf=0.9)
        batch = ObservationBatch(
            robot_id="robot_001",
            intersection_id="a",
            observations=[frame] * config.MAX_OBSERVATIONS_PER_BATCH,
        )
        self.assertEqual(len(batch.observations), config.MAX_OBSERVATIONS_PER_BATCH)


class ArrivalReportLimitsTest(unittest.TestCase):
    def base_kwargs(self, **overrides):
        kwargs = dict(
            robot_id="robot_001",
            intersection_id="seongsu_station",
            arrival_time=0.0,
            depart_time=1.0,
        )
        kwargs.update(overrides)
        return kwargs

    def test_default_transitions_is_an_empty_list(self):
        report = ArrivalReport(**self.base_kwargs())
        self.assertEqual(report.transitions, [])

    def test_empty_ids_are_rejected(self):
        with self.assertRaises(ValidationError):
            ArrivalReport(**self.base_kwargs(robot_id=""))
        with self.assertRaises(ValidationError):
            ArrivalReport(**self.base_kwargs(intersection_id=""))

    def test_oversized_transitions_are_rejected(self):
        transition = dict(timestamp=0.0, from_color="RED", to_color="GREEN")
        with self.assertRaises(ValidationError):
            ArrivalReport(
                **self.base_kwargs(
                    transitions=[transition] * (config.MAX_TRANSITIONS_PER_ARRIVAL + 1)
                )
            )

    def test_transitions_at_the_limit_are_accepted(self):
        transition = dict(timestamp=0.0, from_color="RED", to_color="GREEN")
        report = ArrivalReport(
            **self.base_kwargs(transitions=[transition] * config.MAX_TRANSITIONS_PER_ARRIVAL)
        )
        self.assertEqual(len(report.transitions), config.MAX_TRANSITIONS_PER_ARRIVAL)


class TravelReportLimitsTest(unittest.TestCase):
    def test_overlong_destination_id_is_rejected(self):
        with self.assertRaises(ValidationError):
            TravelReport(
                robot_id="robot_001",
                destination_id="x" * (config.MAX_ID_LENGTH + 1),
                started_at=0.0,
                arrives_at=1.0,
            )

    def test_normal_report_is_accepted(self):
        report = TravelReport(
            robot_id="robot_001",
            destination_id="seongsu_station",
            started_at=0.0,
            arrives_at=1.0,
        )
        self.assertEqual(report.destination_id, "seongsu_station")


if __name__ == "__main__":
    unittest.main()
