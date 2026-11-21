import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from gnn.run_full_auto import process


class FullAutoTests(unittest.TestCase):
    def test_stream_and_cohort_report(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root=Path(temp)
            source=root/"usage.csv"
            with source.open("w",encoding="utf-8") as f:
                f.write("container_id,machine_id,time_stamp,cpu_util_percent,mem_util_percent\n")
                for t in range(80):
                    for i in range(4):
                        f.write(f"c_{i},m_1,{t*1200},{i+1+t%4},{20+i+t%3}\n")
            report=process(source,root/"out",max_cohorts=1,seeds=(11,))
            self.assertEqual(report["evaluated_cohorts"],1)
            self.assertEqual(report["distinct_containers"],4)
            self.assertEqual(report["evaluated_container_ids"],4)
            self.assertTrue((root/"out"/"report.json").exists())
            resumed=process(source,root/"out",max_cohorts=1,seeds=(11,))
            self.assertEqual(resumed["input_rows_scanned"],report["input_rows_scanned"])
