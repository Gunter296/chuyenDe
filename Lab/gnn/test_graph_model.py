import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_model import build, graph
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "host_selection"))
from train_rnn import build as build_rnn


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.prepared = {
            "containers": ["a", "b", "c"],
            "bucket_seconds": 300,
            "actual": [
                {"a": [3+t%7, 2+t%5],
                 "b": [4+t%7, 3+t%5],
                 "c": [8-(t%7), 4+t%3]}
                for t in range(80)]
        }

    def test_edges_use_train_only(self):
        a, _ = graph(self.prepared["actual"], self.prepared["containers"], 48)
        altered = [dict(x) for x in self.prepared["actual"]]
        for row in altered[48:]:
            row["b"] = [1000, 1000]
        b, _ = graph(altered, self.prepared["containers"], 48)
        self.assertTrue((a == b).all())

    def test_graph_and_ablation_run(self):
        graph_case = build(self.prepared, hidden=4, epochs=2)
        no_graph = build(self.prepared, hidden=4, epochs=2, graph_enabled=False)
        self.assertGreater(graph_case["metadata"]["edge_count"], 0)
        self.assertEqual(no_graph["metadata"]["edge_count"], 0)
        self.assertEqual(len(graph_case["forecast"]), len(no_graph["forecast"]))

    def test_same_test_window_as_rnn_and_no_future_leakage(self):
        gnn = build(self.prepared, hidden=4, epochs=2)
        rnn = build_rnn(self.prepared, history=6, horizon=1,
                        hidden=4, epochs=2)
        self.assertEqual(gnn["actual"], rnn["actual"])
        altered = {**self.prepared,
                   "actual": [dict(row) for row in self.prepared["actual"]]}
        altered["actual"][-1]["a"] = [1000, 1000]
        changed = build(altered, hidden=4, epochs=2)
        self.assertEqual(gnn["forecast"][0], changed["forecast"][0])
