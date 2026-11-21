import json
import unittest
from pathlib import Path
from gnn.run_80_20 import fit


class ComparableSplitTests(unittest.TestCase):
    def test_real_subset_split_and_scale(self):
        path = Path(__file__).resolve().parents[1] / "host_selection" / "prepared_shared.json"
        prepared = json.loads(path.read_text(encoding="utf-8"))
        result = fit(prepared, seed=11, use_graph=True, epochs=3)
        self.assertEqual(result["split"]["development_end"], int(len(prepared["actual"]) * .8))
        self.assertEqual(result["sample_counts"], {"train": 5, "validation": 4, "test": 5})
        self.assertEqual(result["test_times"][0] >= result["split"]["development_end"], True)
        self.assertTrue(all(0 <= value <= 1 for row in result["test_truth"] for value in row))
