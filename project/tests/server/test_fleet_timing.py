"""Live uploads must catch up after delays without changing detector cadence."""

import unittest
from unittest.mock import patch

from crossphase_miner.core.models import SignalColor
from server import config
from server.fleet import Robot


class FleetTimingTest(unittest.TestCase):
    def test_delayed_upload_catches_up_without_skipping_or_future_frames(self):
        class Clock:
            current = config.default_sim_start()

            def now(self):
                return self.current

            confirmed_now = now

            def sleep_until(self, target):
                self.current = max(self.current, target)

        clock = Clock()
        batches = []

        class Client:
            def get(self, path):
                return {"model": None}

            def post(self, path, payload):
                if path == "/v1/observations":
                    batches.append((clock.now(), payload["observations"]))
                    if len(batches) == 1:
                        # Server/network processing consumed 4.5 simulated seconds.
                        clock.current += 4.5
                return {}

        class World:
            def color_at(self, crossing, timestamp):
                return SignalColor.RED

        robot = Robot(
            "robot_001",
            Client(),
            clock,
            World(),
            ["test"],
            [],
            1,
            obs_rate_hz=15,
            batch_seconds=1,
            misclass_prob=0,
        )
        with (
            patch.object(config, "TRAVEL_SECONDS_RANGE", (0, 0)),
            patch("server.fleet.SCOUT_TIMEOUT_SECONDS", 12),
        ):
            robot.run_arrival()
        self.assertGreater(len(batches), 2)
        self.assertGreater(len(batches[1][1]), 60)  # one catch-up batch, not stale small batches
        times = [frame["t"] for _, batch in batches for frame in batch]
        for first, second in zip(times, times[1:]):
            self.assertAlmostEqual(second - first, 1 / 15, places=6)
        for sent_at, frames in batches:
            self.assertLessEqual(max(f["t"] for f in frames), sent_at)
            self.assertLess(sent_at - frames[-1]["t"], 0.07)


if __name__ == "__main__":
    unittest.main()
