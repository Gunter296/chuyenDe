"""Resumable all-cohort evaluation using a disk-backed Alibaba trace index."""
import csv
import hashlib
import json
import math
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from alibaba_hosts import load_hosts
from scheduler import simulate
from run import fit, metrics, predict, windows


def say(message):
    print(message, file=sys.stderr, flush=True)


def database_for(args):
    db = args.database or args.output / 'usage.sqlite'
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('CREATE TABLE IF NOT EXISTS usage ('
                'container TEXT, bucket INTEGER, cpu REAL, ram REAL, '
                'PRIMARY KEY(container,bucket))')
    con.execute('CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT)')
    source = args.input.resolve()
    stat = source.stat()
    signature = json.dumps([str(source), stat.st_size, stat.st_mtime_ns,
                            args.bucket_seconds])
    state = dict(con.execute('SELECT key,value FROM state'))
    if state and state.get('signature') != signature:
        raise ValueError('SQLite index belongs to a different CSV or bucket size; use another --database')
    offset = int(state.get('offset', 0))
    if offset == stat.st_size:
        con.execute('CREATE INDEX IF NOT EXISTS usage_bucket ON usage(bucket)')
        say(f'[index] reusing complete database: {db}')
        return con, db
    say(f'[index] ingesting CSV from byte {offset:,}/{stat.st_size:,}')
    batch = []
    valid_rows = int(state.get('valid_rows', state.get('valid', 0)))
    with source.open('rb') as file:
        file.seek(offset)
        while raw := file.readline():
            try:
                row = next(csv.reader([raw.decode('utf-8')]))
                if len(row) < 5 or not row[0].startswith('c_'):
                    continue
                bucket = int(float(row[2]) // args.bucket_seconds)
                cpu, ram = float(row[3]), float(row[4])
                if not (0 <= cpu <= 100 and 0 <= ram <= 100):
                    continue
                batch.append((row[0], bucket, cpu, ram))
                valid_rows += 1
            except (ValueError, UnicodeDecodeError, csv.Error):
                continue
            if len(batch) >= 10000:
                con.executemany('INSERT INTO usage VALUES (?,?,?,?) '
                    'ON CONFLICT(container,bucket) DO UPDATE SET '
                    'cpu=max(cpu,excluded.cpu),ram=max(ram,excluded.ram)', batch)
                con.executemany('INSERT OR REPLACE INTO state VALUES (?,?)',
                    [('signature', signature), ('offset', str(file.tell())),
                     ('valid_rows', str(valid_rows))])
                con.commit()
                batch.clear()
                if file.tell() // 100000000 > offset // 100000000:
                    offset = file.tell()
                    say(f'[index] {100*offset/stat.st_size:.1f}%')
        if batch:
            con.executemany('INSERT INTO usage VALUES (?,?,?,?) '
                'ON CONFLICT(container,bucket) DO UPDATE SET '
                'cpu=max(cpu,excluded.cpu),ram=max(ram,excluded.ram)', batch)
        con.executemany('INSERT OR REPLACE INTO state VALUES (?,?)',
            [('signature', signature), ('offset', str(stat.st_size)),
             ('valid_rows', str(valid_rows))])
        con.commit()
    con.execute('CREATE INDEX IF NOT EXISTS usage_bucket ON usage(bucket)')
    say('[index] complete')
    return con, db


def candidate_arrays(con, start, length, cohort_size, history, horizon,
                     min_train_coverage):
    train_end = int(length*.6)
    rows = con.execute('SELECT container,bucket,cpu,ram FROM usage '
                       'WHERE bucket>=? AND bucket<? ORDER BY container,bucket',
                       (start, start+length))
    by_id = defaultdict(dict)
    for cid, bucket, cpu, ram in rows:
        by_id[cid][bucket-start] = (cpu/100, ram/100)
    eligible = []
    for cid, values in by_id.items():
        coverage = sum(t < train_end for t in values) / train_end
        if coverage >= min_train_coverage:
            eligible.append((cid, coverage, values))
    eligible.sort(key=lambda item: (-item[1], item[0]))
    arrays = {}
    for cid, _, values in eligible:
        a = np.full((length, 2), np.nan, dtype=np.float32)
        for t, v in values.items():
            a[t] = v
        arrays[cid] = a
    return set(by_id), arrays


def complete_mask(a, history, horizon):
    cuts = (int(len(a)*.6), int(len(a)*.8), len(a))
    good = np.isfinite(a).all(axis=1)
    masks = []
    for start, end in zip((0, cuts[0], cuts[1]), cuts):
        masks.append(np.asarray([good[t-history:t+horizon].all()
            for t in range(max(history, start), end-horizon+1)], dtype=bool))
    return masks


def cohorts(arrays, size, history, horizon, min_test, skipped):
    masks = {cid: complete_mask(a, history, horizon) for cid, a in arrays.items()}
    remaining = list(arrays)
    while remaining:
        anchor = remaining.pop(0)
        group = [anchor]
        joint = masks[anchor]
        if not all(x.any() for x in joint[:2]) or joint[2].sum() < min_test:
            skipped['insufficient_windows'] += 1
            continue
        for cid in list(remaining):
            trial = [a & b for a, b in zip(joint, masks[cid])]
            if all(x.any() for x in trial[:2]) and trial[2].sum() >= min_test:
                group.append(cid)
                joint = trial
                remaining.remove(cid)
                if len(group) == size:
                    break
        if len(group) == size:
            yield group
        else:
            skipped['cannot_form_cohort'] += len(group)


def evaluate(a, args, capacity, background):
    parts, cuts = windows(a, args.history, args.horizon)
    train, val, test = parts
    # The simulator advances one bucket per decision; never compress gaps.
    runs = []
    begin = 0
    for i in range(1, len(test)+1):
        if i == len(test) or test[i][0] != test[i-1][0]+1:
            runs.append(test[begin:i])
            begin = i
    test = max(runs, key=len)
    if len(test) < args.min_test_windows:
        raise ValueError('too few test windows')
    parts[2] = test
    scale = np.maximum(np.nanquantile(a[:cuts[0]], .99, axis=(0, 1)), .01).astype(np.float32)
    truth_val = np.asarray([y for _, _, y in val])
    truth_test = np.asarray([y for _, _, y in test])
    predictions = {}
    model_cost = {}
    for name, mode in [('RNN', 'rnn'), ('GNN', 'gnn')]:
        net, seconds = fit(mode, train, val, scale, args.epochs,
                           args.hidden, args.seed, args.horizon)
        pv, _ = predict(net, val, scale)
        pt, inference_ms = predict(net, test, scale)
        margin = (np.maximum(np.quantile(truth_val-pv, .95, axis=(0, 1)), 0)
                  if name == 'RNN' else None)
        predictions[name] = (pt, margin)
        model_cost[name] = {'training_seconds': seconds,
                            'inference_ms_per_step': inference_ms}
    results = {
        'rnn_host_selection': {'prediction': metrics(predictions['RNN'][0], truth_test)},
        'gnn_forecast': {'prediction': metrics(predictions['GNN'][0], truth_test),
                         'system': None}}
    test_background = np.asarray([background[t] for t, _, _ in test])
    pred, margin = predictions['RNN']
    results['rnn_host_selection']['system'] = simulate(
        truth_test[:, :, 0, :], pred + margin[None, None, :, :],
        args.bucket_seconds, capacity, args.alpha, args.beta,
        args.boot_steps, test_background)
    return {'windows': list(map(len, parts)), 'results': results,
            'model_cost': model_cost}


def aggregate(records, exclusions, seen_count, eligible_count, args, complete):
    summary = {'complete': complete, 'cohorts_evaluated': len(records),
               'containers_seen_in_windows': seen_count,
               'containers_meeting_train_coverage': eligible_count,
               'containers_evaluated': len(set(c for r in records for c in r['containers'])),
               'exclusions': dict(exclusions), 'comparison': {},
               'paper_rnn_reference': {'mae': .190, 'rmse': .250,
                   'scope': 'Table 1, RNN(Ours); not directly comparable to this test set'}}
    if not records:
        return summary
    for label in ('rnn_host_selection', 'gnn_forecast'):
        weights = np.asarray([r['windows'][2]*len(r['containers']) for r in records], dtype=float)
        mae = np.asarray([r['results'][label]['prediction']['mae_by_horizon_cpu_ram'] for r in records])
        rmse = np.asarray([r['results'][label]['prediction']['rmse_by_horizon_cpu_ram'] for r in records])
        method = {'mae_by_horizon_cpu_ram': np.average(mae, axis=0, weights=weights).tolist(),
                  'rmse_by_horizon_cpu_ram': np.sqrt(np.average(rmse**2, axis=0, weights=weights)).tolist(),
                  'cohort_mae_std_by_horizon_cpu_ram': np.std(mae, axis=0).tolist(),
                  'test_container_windows': int(weights.sum())}
        if label == 'rnn_host_selection':
            systems = [r['results'][label]['system'] for r in records]
            denominator = sum(s['active_host_steps'] for s in systems)
            method['system'] = {
                'overload_rate': sum(s['overload_rate']*s['active_host_steps'] for s in systems)/denominator,
                'machine_hours_total_across_cohorts': sum(s['machine_hours'] for s in systems),
                'machine_hours_mean_per_cohort': float(np.mean([s['machine_hours'] for s in systems])),
                'machine_hours_std_per_cohort': float(np.std([s['machine_hours'] for s in systems])),
                'migrations_total': sum(s['migrations'] for s in systems),
                'migrations_mean_per_cohort': float(np.mean([s['migrations'] for s in systems])),
                'migrations_std_per_cohort': float(np.std([s['migrations'] for s in systems])),
                'overload_rate_std_per_cohort': float(np.std([s['overload_rate'] for s in systems])),
                'active_host_steps': denominator}
        summary['comparison'][label] = method
    summary['model_cost'] = {}
    for name in ('RNN', 'GNN'):
        summary['model_cost'][name] = {
            'training_seconds_mean': float(np.mean([r['model_cost'][name]['training_seconds'] for r in records])),
            'training_seconds_std': float(np.std([r['model_cost'][name]['training_seconds'] for r in records])),
            'inference_ms_per_step_mean': float(np.mean([r['model_cost'][name]['inference_ms_per_step'] for r in records])),
            'inference_ms_per_step_std': float(np.std([r['model_cost'][name]['inference_ms_per_step'] for r in records]))}
    return summary


def run_all(args):
    args.output.mkdir(parents=True, exist_ok=True)
    con, db = database_for(args)
    lo, hi = con.execute('SELECT min(bucket),max(bucket) FROM usage').fetchone()
    if lo is None:
        raise ValueError('Usage database is empty')
    signature = hashlib.sha256(json.dumps({'schema': 'simple-comparison-v1', **{
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        if k not in ('max_cohorts',)}}, sort_keys=True).encode()).hexdigest()
    checkpoint = args.output / 'comparison_cohorts.jsonl'
    metadata = args.output / 'comparison_config.json'
    if metadata.exists():
        old = json.loads(metadata.read_text())
        if old['signature'] != signature:
            raise ValueError('Output belongs to different settings; use another --output directory')
    else:
        metadata.write_text(json.dumps({'signature': signature, 'source': str(args.input),
            'database': str(db), 'config': {k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()}}, indent=2))
    records = []
    if checkpoint.exists():
        with checkpoint.open(encoding='utf-8') as file:
            records = [json.loads(line) for line in file if line.strip()]
    done = {r['key'] for r in records}
    if args.max_cohorts and len(records) >= args.max_cohorts:
        say(f'[cohort] preview limit already reached: {len(records)}')
        return
    host_count = 2*args.containers
    first_block = lo // args.window_buckets
    last_block = hi // args.window_buckets
    if args.synthetic_hosts:
        capacity = np.full((host_count, 2), args.capacity)
        all_background = np.zeros(((last_block+1)*args.window_buckets-first_block*args.window_buckets,
                                   host_count, 2))
        missing = 0.
    else:
        host_ids, capacity, all_background, missing = load_hosts(
            args.machine_meta, args.machine_usage, first_block*args.window_buckets,
            (last_block-first_block+1)*args.window_buckets,
            args.bucket_seconds, host_count)
        capacity *= args.capacity
    seen, eligible = set(), set()
    exclusions = Counter()
    stopped_early = False
    try:
        for block in range(first_block, last_block+1):
            start = block*args.window_buckets
            observed, arrays = candidate_arrays(con, start, args.window_buckets,
                args.containers, args.history, args.horizon, args.min_train_coverage)
            seen.update(observed)
            eligible.update(arrays)
            exclusions['insufficient_train_coverage_window_instances'] += len(observed)-len(arrays)
            skipped = Counter()
            for group in cohorts(arrays, args.containers, args.history,
                                 args.horizon, args.min_test_windows, skipped):
                key = f'{block}:' + ','.join(group)
                if key in done:
                    continue
                a = np.stack([arrays[c] for c in group], axis=1)
                offset = start-first_block*args.window_buckets
                background = all_background[offset:offset+args.window_buckets]
                try:
                    result = evaluate(a, args, capacity, background)
                except ValueError:
                    exclusions['evaluation_failure'] += 1
                    continue
                record = {'key': key, 'start_bucket': start, 'containers': group, **result}
                with checkpoint.open('a', encoding='utf-8') as file:
                    file.write(json.dumps(record, ensure_ascii=False)+'\n')
                records.append(record); done.add(key)
                say(f'[cohort] {len(records)} done, block {block}, test windows {record["windows"][2]}')
                if args.max_cohorts and len(records) >= args.max_cohorts:
                    stopped_early = True
                    break
            exclusions.update(skipped)
            if stopped_early:
                break
    except KeyboardInterrupt:
        stopped_early = True
        say('[cohort] interrupted; completed cohorts are saved')
    summary = aggregate(records, exclusions, len(seen), len(eligible), args, not stopped_early)
    summary['background_missing_fraction'] = missing
    summary['host_pool'] = ('synthetic' if args.synthetic_hosts else host_ids)
    summary['capacity_cpu_ram'] = capacity.tolist()
    summary['background_interpretation'] = 'empirical scenario added to container demand; not physical core/GB accounting'
    summary['config'] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    state = dict(con.execute('SELECT key,value FROM state'))
    summary['source_valid_rows'] = int(state.get('valid_rows', state.get('valid', 0)))
    target = args.output / 'comparison_full.json'
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    say(f'[report] {target}; complete={summary["complete"]}')
