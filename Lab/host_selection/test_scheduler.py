import unittest
from scheduler import State, schedule
from run import run
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from prepare_trace import prepare
from train_rnn import build


class SchedulerTests(unittest.TestCase):
    def test_rejects_host_with_insufficient_ram(self):
        state = State({"a": (10, 10), "b": (10, 3), "c": (10, 10)},
                      {"x": "a", "y": "a"})
        decision = schedule(state, {"x": [(6, 5)], "y": [(5, 5)]}, 1,
                            boot_minutes=0)
        self.assertEqual(decision.placement["x"], "c")

    def test_all_horizons_checked(self):
        state = State({"a": (10, 10), "b": (10, 10), "c": (10, 10)},
                      {"x": "a", "y": "a", "z": "b"})
        forecast = {"x": [(5, 2), (7, 2)], "y": [(5, 2), (5, 2)],
                    "z": [(1, 1), (5, 1)]}
        decision = schedule(state, forecast, 1, boot_minutes=0)
        self.assertEqual(decision.placement["x"], "c")

    def test_boot_delay_keeps_container_and_reports_unresolved(self):
        state = State({"a": (10, 10)}, {"x": "a", "y": "a"})
        decision = schedule(state, {"x": [(6, 2)], "y": [(6, 2)]},
                            1, boot_minutes=2)
        self.assertEqual(decision.moves, [])
        self.assertEqual(len(decision.opened), 1)
        self.assertEqual(len(decision.unresolved), 1)
        self.assertEqual(state.placement, {"x": "a", "y": "a"})

    def test_stable_load_no_move(self):
        state = State({"a": (10, 10)}, {"x": "a"})
        self.assertEqual(schedule(state, {"x": [(2, 2)]}, 1).moves, [])

    def test_future_actual_does_not_change_earlier_decision(self):
        data = json.loads((Path(__file__).parent / "example.json").read_text())
        first = State({"h1": (10, 10), "h2": (10, 10)},
                      dict(data["placement"]))
        second = State({"h1": (10, 10), "h2": (10, 10)},
                       dict(data["placement"]))
        prediction = {c: [tuple(v) for v in data["forecast"][1][c]]
                      for c in data["placement"]}
        a = schedule(first, prediction, 1, boot_minutes=0)
        data["actual"][2]["a"] = [100, 100]
        b = schedule(second, prediction, 1, boot_minutes=0)
        self.assertEqual(a, b)

    def test_example_comparison(self):
        data = json.loads((Path(__file__).parent / "example.json").read_text())
        self.assertEqual(run(data, "B0")["overloaded_host_steps"], 1)
        self.assertEqual(run(data, "B1")["overloaded_host_steps"], 0)

    def test_prepare_trace_preserves_missing_samples(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            source = Path(tmp) / "usage.csv"
            output = Path(tmp) / "prepared.json"
            source.write_text("a,h1,0,2,3\na,h1,300,4,5\n"
                              "b,h2,0,1,2\nb,h2,600,3,4\n")
            result = prepare(source, output, max_containers=2,
                             min_coverage=0.5)
            self.assertEqual(result["actual"][1], {"a": [4.0, 5.0]})
            self.assertEqual(result["missing_fraction"], 2 / 6)

    def test_train_builds_causal_comparison_input(self):
        actual = [{"a": [2 + (t % 5), 3 + (t % 3)],
                   "b": [4 + (t % 7), 2 + (t % 4)]}
                  for t in range(80)]
        prepared = {"actual": actual, "containers": ["a", "b"],
                    "bucket_seconds": 300}
        data = build(prepared, history=6, horizon=1, hidden=4, epochs=2)
        self.assertGreaterEqual(len(data["actual"]), 2)
        self.assertEqual(len(data["actual"]), len(data["forecast"]))
        self.assertEqual(data["metadata"]["train_end"], 48)
        self.assertIsNotNone(run(data, "B1")["overload_rate"])


if __name__ == "__main__":
    unittest.main()
