"""End-to-end checks for ingestion -> RNN -> scheduler on synthetic data."""
import io
import json
import sys
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from prepare_trace import prepare
from train_rnn import build
from run import run
from experiment import experiment


class PipelineTests(unittest.TestCase):
    def test_tar_archive_and_complete_pipeline(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            csv_bytes = io.BytesIO()
            for t in range(80):
                for c, base in (("a", 4), ("b", 5), ("c", 6)):
                    cpu = base + (t % 7)
                    ram = base + (t % 5)
                    csv_bytes.write(f"{c},h1,{t * 300},{cpu},{ram}\n".encode())
            raw = csv_bytes.getvalue()
            archive = root / "container_usage.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                info = tarfile.TarInfo("container_usage.csv")
                info.size = len(raw)
                tar.addfile(info, io.BytesIO(raw))
            prepared = prepare(archive, root / "prepared.json",
                               max_containers=3)
            self.assertEqual(prepared["missing_fraction"], 0)
            self.assertEqual(len(prepared["actual"]), 80)
            comparison = build(prepared, history=6, horizon=1,
                               hidden=4, epochs=2)
            self.assertEqual(comparison["metadata"]["test_steps"], 16)
            self.assertEqual(len(comparison["metadata"]
                                 ["test_mae_by_horizon_cpu_ram"]), 1)
            result = [run(comparison, mode) for mode in ("B0", "B1")]
            self.assertTrue(all(x["observed_host_steps"] > 0 for x in result))
            self.assertTrue(all(x["active_host_hours"] > 0 for x in result))
            report = experiment(prepared, root / "results", seeds=(11, 22),
                                history=6, horizon=1, hidden=4, epochs=2)
            self.assertEqual(len(report["per_seed"]), 2)
            self.assertTrue((root / "results" / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
