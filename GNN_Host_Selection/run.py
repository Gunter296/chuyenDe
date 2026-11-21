"""Reproducible five-horizon forecast and placement experiment."""
import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scheduler import simulate
from alibaba_hosts import load_hosts, container_machine_ids


def load_trace(path, count, bucket_seconds, history=12, horizon=5,
               window_buckets=72):
    if path.suffix == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
        ids = data['containers'][:count]
        rows = data['actual']
        a = np.full((len(rows), len(ids), 2), np.nan, dtype=np.float32)
        for t, row in enumerate(rows):
            for i, cid in enumerate(ids):
                if cid in row:
                    a[t, i] = row[cid]
        return a / 100, ids, data['bucket_seconds'], data.get('start_bucket', 0)
    counts = defaultdict(lambda: defaultdict(int))
    # Choose a bounded interval using train observations only. Container
    # lifetimes are shorter than the entire trace.
    train_buckets = int(window_buckets * .6)
    with path.open(encoding='utf-8', newline='') as file:
        for row in csv.DictReader(file):
            try:
                if not row['container_id']:
                    continue
                bucket = int(float(row['time_stamp']) // bucket_seconds)
                float(row['cpu_util_percent']); float(row['mem_util_percent'])
            except (ValueError, KeyError):
                continue
            block, offset = divmod(bucket, window_buckets)
            if offset < train_buckets:
                counts[block][row['container_id']] += 1
    if not counts:
        raise ValueError('No valid container records')
    ranked_blocks = sorted(counts, key=lambda b: (
        sorted(counts[b].values(), reverse=True)[min(count-1, len(counts[b])-1)],
        sum(sorted(counts[b].values(), reverse=True)[:count]), -b), reverse=True)
    candidates = {b: sorted(counts[b], key=lambda c: (-counts[b][c], c))[:max(count*10, count)]
                  for b in ranked_blocks}
    lookup = {b: {c: i for i, c in enumerate(ids)} for b, ids in candidates.items()}
    values = {b: defaultdict(dict) for b in ranked_blocks}
    with path.open(encoding='utf-8', newline='') as file:
        for row in csv.DictReader(file):
            try:
                bucket = int(float(row['time_stamp']) // bucket_seconds)
                block = bucket // window_buckets
                i = lookup[block][row['container_id']]
                v = [float(row['cpu_util_percent']), float(row['mem_util_percent'])]
            except (ValueError, KeyError):
                continue
            if min(v) < 0:
                continue
            old = values[block][bucket].get(i, [0., 0.])
            values[block][bucket][i] = np.maximum(old, v)
    best_found = 0
    for block in ranked_blocks:
        ids = candidates[block]
        lo = block * window_buckets
        a = np.full((window_buckets, len(ids), 2), np.nan, dtype=np.float32)
        for bucket, row in values[block].items():
            for i, v in row.items():
                a[bucket-lo, i] = v
        selected = select_cohort(a, count, history, horizon)
        best_found = max(best_found, len(selected))
        if len(selected) >= 2:
            return a[:, selected] / 100, [ids[i] for i in selected], bucket_seconds, lo
    raise ValueError(f'No shared interval with two complete container series; '
                     f'best cohort size={best_found}, intervals checked={len(ranked_blocks)}, '
                     f'bucket_seconds={bucket_seconds}, window_buckets={window_buckets}. '
                     'No test result was produced.')


def select_cohort(a, count, history, horizon):
    ids = list(range(a.shape[1]))
    cuts = (int(len(a)*.6), int(len(a)*.8), len(a))
    masks = []
    for i in range(len(ids)):
        good = np.isfinite(a[:, i]).all(axis=1)
        parts = []
        for start, end in zip((0, cuts[0], cuts[1]), cuts):
            parts.append(np.asarray([good[t-history:t+horizon].all()
                for t in range(max(history, start), end-horizon+1)], dtype=bool))
        masks.append(parts)
    selected = []
    joint = None
    while len(selected) < count:
        choices = []
        for i, parts in enumerate(masks):
            if i in selected:
                continue
            trial = parts if joint is None else [x & y for x, y in zip(joint, parts)]
            sizes = [int(x.sum()) for x in trial]
            if min(sizes) > 0:
                choices.append((min(sizes), sum(sizes), -i, i, trial))
        if not choices:
            break
        _, _, _, i, joint = max(choices, key=lambda x: x[:3])
        selected.append(i)
    return selected


def windows(a, history, horizon):
    n = len(a); cuts = (int(n*.6), int(n*.8), n)
    result = []
    for start, end in zip((0, cuts[0], cuts[1]), cuts):
        samples = []
        for t in range(max(history, start), end-horizon+1):
            x, y = a[t-history:t], a[t:t+horizon]
            if np.isfinite(x).all() and np.isfinite(y).all():
                samples.append((t, x.transpose(1, 0, 2), y.transpose(1, 0, 2)))
        result.append(samples)
    if any(not part for part in result):
        raise ValueError('No complete windows in a time split; choose a denser cohort')
    return result, cuts


def graph(x, top_k=2):
    # x: C×history×2. Correlation uses observed history only.
    c = len(x); flat = x.reshape(c, -1)
    corr = np.nan_to_num(np.corrcoef(flat), nan=0.) if c > 1 else np.zeros((1, 1))
    np.fill_diagonal(corr, 0)
    adj = np.zeros((c, c), dtype=np.float32)
    for i in range(c):
        for j in np.argsort(-corr[i], kind='stable')[:min(top_k, c-1)]:
            if corr[i, j] > 0:
                adj[i, j] = corr[i, j]
    return adj / np.maximum(adj.sum(1, keepdims=True), 1e-8)


class Forecaster(nn.Module):
    def __init__(self, hidden, horizon, mode):
        super().__init__()
        self.mode = mode
        self.encoder = nn.GRU(2, hidden, batch_first=True)
        self.message = nn.Linear(hidden, hidden) if mode == 'gnn' else None
        self.head = nn.Linear(hidden, horizon*2)
        self.horizon = horizon

    def forward(self, x, adj=None):
        h = self.encoder(x)[1][-1]
        if self.message is not None:
            h = torch.relu(h + self.message(adj @ h))
        return self.head(h).reshape(len(x), self.horizon, 2)


def fit(mode, train, val, scale, epochs, hidden, seed, horizon):
    torch.manual_seed(seed)
    net = Forecaster(hidden, horizon, mode)
    opt = torch.optim.Adam(net.parameters(), lr=.003)
    best, weights = float('inf'), None
    started = time.perf_counter()
    for _ in range(epochs):
        net.train()
        for _, x, y in train:
            xt = torch.tensor(x / scale); yt = torch.tensor(y / scale)
            adj = torch.tensor(graph(x)) if mode == 'gnn' else None
            pred = net(xt, adj)
            loss = ((pred-yt)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            losses = []
            for _, x, y in val:
                adj = torch.tensor(graph(x)) if mode == 'gnn' else None
                losses.append(float(((net(torch.tensor(x/scale), adj)-torch.tensor(y/scale))**2).mean()))
        score = float(np.mean(losses))
        if score < best:
            best = score
            weights = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(weights)
    return net, time.perf_counter()-started


def predict(net, samples, scale):
    out = []; elapsed = 0.
    net.eval()
    with torch.no_grad():
        for _, x, _ in samples:
            adj = torch.tensor(graph(x)) if net.mode == 'gnn' else None
            start = time.perf_counter()
            p = net(torch.tensor(x/scale), adj).numpy()*scale
            elapsed += time.perf_counter()-start
            out.append(np.maximum(p, 0))
    return np.asarray(out), elapsed*1000/len(samples)


def metrics(pred, truth):
    e = pred-truth
    return {'mae_by_horizon_cpu_ram': np.mean(np.abs(e), axis=(0, 1)).tolist(),
            'rmse_by_horizon_cpu_ram': np.sqrt(np.mean(e*e, axis=(0, 1))).tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('results'))
    p.add_argument('--containers', type=int, default=4)
    p.add_argument('--bucket-seconds', type=int, default=1200)
    p.add_argument('--window-buckets', type=int, default=72,
                   help='Length of a shared trace interval in buckets')
    p.add_argument('--history', type=int, default=12)
    p.add_argument('--horizon', type=int, default=5)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--hidden', type=int, default=24)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--capacity', type=float, default=1.)
    p.add_argument('--alpha', type=float, default=.15)
    p.add_argument('--beta', type=float, default=.1)
    p.add_argument('--boot-steps', type=int, default=1)
    p.add_argument('--machine-meta', type=Path, default=Path('../Lab/data/machine_meta.csv'))
    p.add_argument('--machine-usage', type=Path, default=Path('../Lab/data/machine_usage_bigger.csv'))
    p.add_argument('--container-meta', type=Path, default=Path('../Lab/data/container_meta.csv'))
    p.add_argument('--synthetic-hosts', action='store_true', help='Disable Alibaba host profile')
    p.add_argument('--all-cohorts', action='store_true',
                   help='Evaluate every eligible non-overlapping cohort')
    p.add_argument('--database', type=Path, help='Reusable SQLite usage database')
    p.add_argument('--min-test-windows', type=int, default=5)
    p.add_argument('--min-train-coverage', type=float, default=.9)
    p.add_argument('--max-cohorts', type=int, default=0,
                   help='Preview limit; 0 means all eligible cohorts')
    args = p.parse_args()
    np.random.seed(args.seed); torch.set_num_threads(1)
    if args.all_cohorts:
        from full_run import run_all
        run_all(args)
        return
    a, ids, bucket_seconds, start_bucket = load_trace(
        args.input, args.containers, args.bucket_seconds, args.history,
        args.horizon, args.window_buckets)
    parts, cuts = windows(a, args.history, args.horizon)
    train, val, test = parts
    scale = np.maximum(np.nanquantile(a[:cuts[0]], .99, axis=(0, 1)), .01).astype(np.float32)
    truth_val = np.asarray([y for _, _, y in val])
    truth_test = np.asarray([y for _, _, y in test])
    host_count = max(len(ids), math.ceil(2 * len(ids)))
    if args.synthetic_hosts:
        host_ids = [f'synthetic_{i}' for i in range(host_count)]
        host_capacity = np.full((host_count, 2), args.capacity)
        background = np.zeros((len(a), host_count, 2))
        background_missing = 0.
    else:
        excluded = container_machine_ids(args.container_meta, ids)
        host_ids, host_capacity, background, background_missing = load_hosts(
            args.machine_meta, args.machine_usage, start_bucket, len(a),
            bucket_seconds, host_count, excluded)
        host_capacity *= args.capacity
    report = {'source': str(args.input), 'start_bucket': start_bucket,
              'start_seconds': start_bucket * bucket_seconds,
              'duration_seconds': len(a) * bucket_seconds,
              'containers': ids, 'missing_fraction': float(np.isnan(a).mean()),
              'split_indices': cuts, 'windows': list(map(len, parts)), 'resource_units': 'source percentage / 100; simulation units',
              'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'hosts': {'ids': host_ids, 'normalized_capacity_cpu_ram': host_capacity.tolist(),
                        'background_missing_fraction': background_missing,
                        'background_interpretation': 'empirical utilization scenario; additive simulation assumption'},
              'comparison': {},
              'paper_rnn_reference': {'mae': .190, 'rmse': .250,
                'scope': 'Table 1, RNN(Ours); different test set and normalization; not directly comparable'}}
    forecasts = {}
    for label, mode in [('RNN', 'rnn'), ('GNN', 'gnn')]:
        net, seconds = fit(mode, train, val, scale, args.epochs, args.hidden, args.seed, args.horizon)
        pv, _ = predict(net, val, scale)
        pt, inference_ms = predict(net, test, scale)
        margin = (np.maximum(np.quantile(truth_val-pv, .95, axis=(0, 1)), 0)
                  if label == 'RNN' else None)
        forecasts[label] = (pt, margin)
        report['comparison']['gnn_forecast' if label == 'GNN' else 'rnn_host_selection'] = {
            'prediction': metrics(pt, truth_test),
            'training_seconds': seconds, 'inference_ms_per_step': inference_ms,
            'system': None}
    pred, margin = forecasts['RNN']
    test_background = np.asarray([background[t] for t, _, _ in test])
    report['comparison']['rnn_host_selection']['system'] = simulate(
        truth_test[:, :, 0, :], pred + margin[None, None, :, :], bucket_seconds,
        host_capacity, args.alpha, args.beta, args.boot_steps, test_background)
    report['comparison']['rnn_host_selection']['safety_margin'] = margin.tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'comparison.json'
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
