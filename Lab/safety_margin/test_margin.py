import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from margin import calibrate, apply_margin, empirical_coverage


class MarginTests(unittest.TestCase):
    def test_one_sided_margin_and_no_mutation(self):
        errors = [[[-3, 1]], [[-2, 4]], [[-1, 5]]]
        margin = calibrate(errors, 0.9)
        self.assertEqual(margin[0, 0], 0)
        self.assertGreater(margin[0, 1], 4)
        data = {"calibration_errors": errors,
                "forecast": [{"a": [[2, 3]]}]}
        adjusted = apply_margin(data, 0.9)
        self.assertEqual(data["forecast"][0]["a"], [[2, 3]])
        self.assertEqual(adjusted["forecast"][0]["a"][0][0], 2)

    def test_bad_quantile(self):
        with self.assertRaises(ValueError):
            calibrate([[[1, 1]]], 1)

    def test_coverage_uses_matching_horizon(self):
        data = {
            "actual": [{"a": [2, 3]}, {"a": [4, 5]},
                       {"a": [6, 7]}],
            "forecast": [{"a": [[2, 2], [5, 5]]},
                         {"a": [[3, 6], [7, 7]]},
                         {"a": [[6, 6], [0, 0]]}],
        }
        self.assertEqual(empirical_coverage(data), [[0.5, 0.5], [1.0, 1.0]])
