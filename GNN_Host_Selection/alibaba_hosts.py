"""Read Alibaba machine metadata and an empirical background-load profile.

The usage percentages are used as a scenario profile. They are not a
physical decomposition of machine usage into selected containers and others.
"""
import csv
import numpy as np


def load_hosts(meta_path, usage_path, start_bucket, steps, bucket_seconds,
               host_count, excluded=()):
    excluded = set(excluded)
    machines = {}
    with meta_path.open(encoding='utf-8', newline='') as file:
        for row in csv.reader(file):
            if len(row) < 7 or row[0] in excluded or row[0] in machines:
                continue
            try:
                cpu, mem = float(row[4]), float(row[5])
            except ValueError:
                continue
            if cpu > 0 and mem > 0 and row[6] == 'USING':
                machines[row[0]] = (cpu, mem)
    if len(machines) < host_count:
        raise ValueError('Too few eligible machines in machine_meta.csv')
    ids = sorted(machines)[:host_count]
    index = {name: i for i, name in enumerate(ids)}
    background = np.full((steps, host_count, 2), np.nan, dtype=np.float32)
    with usage_path.open(encoding='utf-8', newline='') as file:
        for row in csv.DictReader(file):
            i = index.get(row['machine_id'])
            if i is None:
                continue
            try:
                t = int(float(row['time_stamp']) // bucket_seconds) - start_bucket
                value = [float(row['cpu_util_percent']), float(row['mem_util_percent'])]
            except ValueError:
                continue
            if 0 <= t < steps and min(value) >= 0:
                old = background[t, i]
                background[t, i] = np.maximum(np.nan_to_num(old), value)
    # Fill only from earlier observations. Remaining gaps use zero and are reported.
    missing = float(np.isnan(background).any(axis=2).mean())
    for t in range(1, steps):
        mask = np.isnan(background[t])
        background[t][mask] = background[t-1][mask]
    background = np.nan_to_num(background) / 100
    physical = np.asarray([machines[name] for name in ids], dtype=np.float32)
    normalized = physical / np.median(physical, axis=0)
    return ids, normalized, background, missing


def container_machine_ids(path, selected):
    selected = set(selected)
    found = set()
    with path.open(encoding='utf-8', newline='') as file:
        for row in csv.reader(file):
            if len(row) >= 2 and row[0] in selected:
                found.add(row[1])
    return found
