import json
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"host_selection"))
from cohort_baseline import fit_rnn, sums_for
from gnn.forecast_core import multiresource_graph
from gnn.run_80_20 import fit


class FutureWorkTests(unittest.TestCase):
    def test_rnn_graph_test_alignment(self):
        source=Path(__file__).resolve().parents[1]/"host_selection"/"prepared_shared.json"
        data=json.loads(source.read_text(encoding="utf-8"))
        graph=multiresource_graph(data["actual"],data["containers"],25)
        rnn=fit_rnn(data,11,epochs=2)
        gnn=fit(data,11,True,epochs=2,adjacency_override=graph,return_full=True)
        self.assertEqual(rnn["test_times"],gnn["test_times"])
        self.assertEqual(sums_for(rnn)["n"],sums_for(gnn)["n"])
        self.assertEqual(graph.shape,(4,4))
