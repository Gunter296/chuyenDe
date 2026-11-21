import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from full_run import cohorts, complete_mask


class CohortTests(unittest.TestCase):
    def test_requires_shared_test_windows(self):
        dense = np.ones((72, 2), dtype=np.float32)
        sparse = dense.copy()
        sparse[57:] = np.nan
        arrays = {f'c{i}': dense.copy() for i in range(4)}
        arrays['missing_test'] = sparse
        skipped = Counter()
        groups = list(cohorts(arrays, 4, 12, 5, 5, skipped))
        self.assertEqual(len(groups), 1)
        self.assertEqual(set(groups[0]), {'c0', 'c1', 'c2', 'c3'})
        self.assertGreater(skipped['insufficient_windows'], 0)

    def test_no_test_labels_means_no_valid_windows(self):
        series = np.ones((72, 2), dtype=np.float32)
        series[57:] = np.nan
        self.assertEqual(int(complete_mask(series, 12, 5)[2].sum()), 0)


if __name__ == '__main__':
    unittest.main()
