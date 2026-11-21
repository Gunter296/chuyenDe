"""Stream Alibaba 2018 container_usage.csv into a compact aligned JSON trace.

The official column order is container_id,machine_id,time_stamp,
cpu_util_percent,mem_util_percent,... . CPU/memory percentages are treated
as normalized simulation loads, not physical cores or GiB.
"""
import argparse
import csv
import json
import gzip
import io
import tarfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def rows(path):
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            member = next((m for m in archive if m.isfile()
                           and m.name.endswith("container_usage.csv")), None)
            if member is None:
                raise ValueError("Archive lacks container_usage.csv")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Cannot read archive member")
            with io.TextIOWrapper(stream, encoding="utf-8", newline="") as f:
                yield csv.reader(f)
    elif path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            yield csv.reader(f)
    else:
        with path.open(newline="", encoding="utf-8") as f:
            yield csv.reader(f)


def prepare(path: Path, output: Path, max_containers=200, min_coverage=0.9,
            window_start_seconds=None, window_duration_seconds=None,
            bucket_seconds=300):
    # Two passes keep memory bounded by the selected container count.
    counts = defaultdict(set)
    candidate_limit = max_containers * 5
    min_bucket, max_bucket = None, None
    with rows(path) as input_rows:
        for row in input_rows:
            if len(row) < 5 or not row[0] or not row[2].replace(".", "", 1).isdigit():
                continue
            try:
                timestamp = float(row[2])
                if window_start_seconds is not None and not (
                        window_start_seconds <= timestamp <
                        window_start_seconds + window_duration_seconds):
                    continue
                bucket = int(timestamp // bucket_seconds)
                float(row[3]); float(row[4])
            except ValueError:
                continue
            if window_start_seconds is not None or row[0] in counts or len(counts) < candidate_limit:
                counts[row[0]].add(bucket)
            min_bucket = bucket if min_bucket is None else min(min_bucket, bucket)
            max_bucket = bucket if max_bucket is None else max(max_bucket, bucket)
    if min_bucket is None:
        raise ValueError("No valid records")
    if window_start_seconds is not None:
        min_bucket = window_start_seconds // bucket_seconds
        max_bucket = (window_start_seconds + window_duration_seconds - 1) // bucket_seconds
    length = max_bucket - min_bucket + 1
    train_length = max(1, int(length * 0.6))
    train_end = min_bucket + train_length
    coverage = {c: sum(min_bucket <= t < train_end for t in times) / train_length
                for c, times in counts.items()}
    selected = sorted((c for c in counts if coverage[c] >= min_coverage),
                      key=lambda c: (-coverage[c], c))[:max_containers]
    if not selected:
        top = sorted(coverage.items(), key=lambda item: -item[1])[:10]
        raise ValueError(f"No containers meet coverage threshold; best={top}")
    selected_set = set(selected)
    records = defaultdict(dict)
    with rows(path) as input_rows:
        for row in input_rows:
            if len(row) < 5 or row[0] not in selected_set:
                continue
            try:
                timestamp = float(row[2])
                if window_start_seconds is not None and not (
                        window_start_seconds <= timestamp <
                        window_start_seconds + window_duration_seconds):
                    continue
                bucket = int(timestamp // bucket_seconds)
                cpu, ram = float(row[3]), float(row[4])
            except ValueError:
                continue
            if cpu < 0 or ram < 0:
                continue
            old = records[bucket].get(row[0], (0, 0))
            records[bucket][row[0]] = [max(old[0], cpu), max(old[1], ram)]
    timeline = []
    missing = 0
    for bucket in range(min_bucket, max_bucket + 1):
        row = {}
        for c in selected:
            value = records[bucket].get(c)
            if value is None:
                missing += 1
            else:
                row[c] = value
        timeline.append(row)
    result = {"bucket_seconds": bucket_seconds, "start_bucket": min_bucket,
              "containers": selected, "actual": timeline,
              "missing_fraction": missing / (length * len(selected)),
              "source": str(path), "resource_units": "Alibaba container usage percentage, simulation units",
              "window_start_seconds": window_start_seconds,
              "window_duration_seconds": window_duration_seconds,
              "min_train_coverage": min_coverage,
              "selected_train_coverage": {c: coverage[c] for c in selected}}
    output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-containers", type=int, default=200)
    parser.add_argument("--window-start-seconds", type=int)
    parser.add_argument("--window-duration-seconds", type=int)
    parser.add_argument("--bucket-seconds", type=int, default=300)
    args = parser.parse_args()
    result = prepare(args.input, args.output, args.max_containers,
                     window_start_seconds=args.window_start_seconds,
                     window_duration_seconds=args.window_duration_seconds,
                     bucket_seconds=args.bucket_seconds)
    print(json.dumps({k: result[k] for k in
                      ("bucket_seconds", "containers", "missing_fraction")},
                     ensure_ascii=False))
