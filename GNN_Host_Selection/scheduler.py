"""Deterministic cost-aware best-fit over a five-step CPU/RAM forecast."""
import math
import numpy as np


def choose(forecast, placement, powered, capacity=1.0, alpha=0.15, beta=0.1):
    """Return placement, migration count, new boot requests.

    A host must fit every horizon/resource. A safe current placement gets its
    own candidate score; beta penalizes moving and alpha penalizes booting.
    """
    f = np.asarray(forecast, dtype=float)
    if f.ndim != 3 or f.shape[2] != 2 or len(placement) != len(f):
        raise ValueError('forecast shape must be containers × horizon × 2')
    cap = np.broadcast_to(np.asarray(capacity, dtype=float), (len(powered), 2))
    if np.any(~np.isfinite(f)) or np.any(f < 0) or np.any(cap <= 0):
        raise ValueError('invalid demand or capacity')
    if any(h < 0 or h >= len(powered) for h in placement):
        raise ValueError('invalid placement')
    result = list(placement)
    active = list(powered)
    loads = np.zeros((len(active), f.shape[1], 2))
    for i, h in enumerate(result):
        loads[h] += f[i]
    moves = boots = 0
    order = sorted(range(len(f)), key=lambda i: (-float(np.max(f[i])), i))
    for i in order:
        old = result[i]
        loads[old] -= f[i]
        candidates = []
        for h in range(len(active)):
            if np.any(loads[h] + f[i] > cap[h] + 1e-9):
                continue
            slack = float(np.mean((cap[h] - loads[h] - f[i]) / cap[h]))
            penalty = (alpha if not active[h] else 0) + (beta if h != old else 0)
            candidates.append((slack + penalty, h))
        if candidates:
            target = min(candidates)[1]
        else:
            # Keep overloaded placement and let the simulator record the failure.
            target = old
        if target != old:
            moves += 1
            if not active[target]:
                active[target] = True
                boots += 1
        result[i] = target
        loads[target] += f[i]
    return result, moves, boots


def simulate(actual, forecast, bucket_seconds, capacity=1.0, alpha=.15,
             beta=.1, boot_steps=1, background=None):
    """actual: T×C×2, forecast: T×C×H×2; no future actual in decisions."""
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    t_count, c_count, _ = actual.shape
    hosts = max(c_count, math.ceil(2 * c_count))
    cap = np.broadcast_to(np.asarray(capacity, dtype=float), (hosts, 2))
    if background is None:
        background = np.zeros((t_count, hosts, 2))
    background = np.asarray(background)
    if background.shape != (t_count, hosts, 2):
        raise ValueError('background shape must be steps × hosts × 2')
    placement = list(range(c_count))
    ready = [0] * c_count + [10**12] * (hosts - c_count)
    overload = active_minutes = migrations = boots = 0
    for t in range(t_count):
        powered = [x <= t for x in ready]
        previous = list(placement)
        # Current background is observable; no future background is supplied.
        available = np.maximum(cap - background[t], 1e-8)
        proposed, _, _ = choose(forecast[t], placement, powered,
                                available, alpha, beta)
        new_boots = 0
        for h in set(proposed):
            if ready[h] == 10**12:
                ready[h] = t + boot_steps
                new_boots += 1
        # Booting hosts cannot serve load until ready. Retain old placement.
        for i, h in enumerate(proposed):
            if ready[h] <= t:
                placement[i] = h
        migrations += sum(a != b for a, b in zip(previous, placement))
        loads = np.zeros((hosts, 2))
        for i, h in enumerate(placement):
            loads[h] += actual[t, i]
        occupied = set(placement)
        for h in range(hosts):
            if h not in occupied and ready[h] <= t:
                ready[h] = 10**12
        overload += int(np.sum(np.any(loads + background[t] > cap + 1e-9, axis=1)))
        active_minutes += sum(x <= t for x in ready)
        boots += new_boots
    return {'overload_rate': overload / max(active_minutes, 1),
            'machine_hours': active_minutes * bucket_seconds / 3600,
            'migrations': migrations, 'boot_requests': boots,
            'active_host_steps': active_minutes}
