import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scheduler import choose, simulate


class SchedulerTests(unittest.TestCase):
    def test_checks_both_resources_all_horizons(self):
        forecast = [[[.1, .8]] * 4 + [[.1, .95]], [[.1, .1]] * 5]
        place, _, _ = choose(forecast, [0, 1], [True, True, True], 1.0, 0, 0)
        self.assertNotEqual(place[0], place[1])

    def test_migration_penalty_keeps_safe_container(self):
        forecast = [[[.2, .2]] * 5, [[.2, .2]] * 5]
        place, moves, _ = choose(forecast, [0, 1], [True, True], 1.0, 0, 10)
        self.assertEqual(place, [0, 1])
        self.assertEqual(moves, 0)

    def test_background_can_cause_overload(self):
        actual = [[[.3, .3]]]
        forecast = [[[[.3, .3]] * 5]]
        background = [[[.8, .8], [0, 0]]]
        result = simulate(actual, forecast, 300, background=background)
        self.assertGreater(result['overload_rate'], 0)


if __name__ == '__main__':
    unittest.main()
