"""Speed changes must preserve time and wake robots already in transit."""

import unittest
from unittest.mock import patch

from pydantic import ValidationError

from server.app import ClockSpeedIn
from server.fleet import Clock
from server.service import SimClock


class SimClockTest(unittest.TestCase):
    def test_all_speeds_preserve_time_and_change_only_future_rate(self):
        wall = [1000.0]
        with patch("time.time", side_effect=lambda: wall[0]):
            clock = SimClock(5000.0, 20)
            wall[0] += 10
            self.assertEqual(clock.now(), 5200)
            for speed in (1, 2, 5, 10, 20, 1):
                before = clock.now()
                result = clock.set_speed(speed)
                self.assertEqual(result["sim_now"], before)
                self.assertEqual(result["speed"], speed)
                wall[0] += 2
                self.assertEqual(clock.now(), before + 2 * speed)

    def test_invalid_speeds_do_not_change_clock(self):
        with patch("time.time", return_value=1000):
            clock = SimClock(5000, 20)
            before = clock.as_dict()
            for speed in (0, -1, 3, 50, 100, float("inf"), float("nan")):
                with self.assertRaises(ValueError):
                    clock.set_speed(speed)
                with self.assertRaises(ValidationError):
                    ClockSpeedIn(speed=speed)
                self.assertEqual(clock.as_dict(), before)

    def test_wait_in_progress_adapts_to_acceleration(self):
        self.check_wait(initial_speed=1, new_speed=20, target=10, expected_wall=0.1 + 9.9 / 20)

    def test_wait_in_progress_adapts_to_deceleration(self):
        self.check_wait(initial_speed=20, new_speed=1, target=6, expected_wall=5.525)

    def check_wait(self, initial_speed, new_speed, target, expected_wall):
        wall = [0.0]
        sleeps = []
        clock = Clock(sim_start=0, wall_start=0, speed=initial_speed)

        def sleep(seconds):
            sleeps.append(seconds)
            wall[0] += seconds
            if len(sleeps) == 1:
                clock.synchronize({"sim_now": wall[0] * initial_speed, "speed": new_speed})
            else:
                clock.synchronize(
                    {
                        "sim_now": sleeps[0] * initial_speed + (wall[0] - sleeps[0]) * new_speed,
                        "speed": new_speed,
                    }
                )

        with (
            patch("time.time", side_effect=lambda: wall[0]),
            patch("time.sleep", side_effect=sleep),
        ):
            clock.sleep_until(target)
            self.assertAlmostEqual(wall[0], expected_wall)
            self.assertAlmostEqual(clock.now(), target)
            self.assertTrue(all(seconds <= 0.1 for seconds in sleeps))

    def test_fast_estimate_cannot_release_observations_before_confirmation(self):
        wall = [0.0]
        clock = Clock(sim_start=0, wall_start=0, speed=20)
        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            wall[0] += seconds
            # Server slowed to 1x at wall time zero. Delay discovery until
            # the second sleep, when the old estimate already passed target.
            if len(sleeps) >= 2:
                clock.synchronize({"sim_now": wall[0], "speed": 1})

        with (
            patch("time.time", side_effect=lambda: wall[0]),
            patch("time.sleep", side_effect=sleep),
        ):
            clock.synchronize({"sim_now": 0, "speed": 20})
            clock.sleep_until(0.1)
            self.assertGreaterEqual(wall[0], 0.1)
            self.assertGreaterEqual(clock._confirmed_time, 0.1)


if __name__ == "__main__":
    unittest.main()
