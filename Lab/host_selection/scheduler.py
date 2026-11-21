"""Deterministic forecast-driven CPU/RAM host placement.

Inputs are simulation resource units. A forecast is container -> list of
(cpu, ram), one pair per future scheduling step. No future actual load is read here.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

Demand = Tuple[float, float]
Forecast = Dict[str, List[Demand]]


@dataclass
class State:
    capacity: Dict[str, Demand]
    placement: Dict[str, str]
    ready_at: Dict[str, int] = field(default_factory=dict)
    last_move: Dict[str, int] = field(default_factory=dict)
    next_host: int = 0


@dataclass
class Decision:
    placement: Dict[str, str]
    moves: List[Tuple[str, str, str]]
    opened: List[str]
    unresolved: List[str]


def _loads(placement: Dict[str, str], forecast: Forecast, hosts: Dict[str, Demand]):
    horizon = len(next(iter(forecast.values())))
    loads = {host: [[0.0, 0.0] for _ in range(horizon)] for host in hosts}
    for container, host in placement.items():
        for h, demand in enumerate(forecast[container]):
            loads[host][h][0] += demand[0]
            loads[host][h][1] += demand[1]
    return loads


def schedule(state: State, forecast: Forecast, minute: int, threshold: float = 0.9,
             boot_minutes: int = 1, move_limit: int | None = None,
             cooldown: int = 5) -> Decision:
    if not forecast or set(forecast) != set(state.placement):
        raise ValueError("Forecast must contain exactly the placed containers")
    horizon = len(next(iter(forecast.values())))
    if horizon == 0 or any(len(v) != horizon or any(
            len(d) != 2 or any(x < 0 for x in d) for d in v) for v in forecast.values()):
        raise ValueError("Forecast must have equal, nonnegative CPU/RAM horizons")
    if not 0 < threshold <= 1 or boot_minutes < 0 or cooldown < 0:
        raise ValueError("Invalid scheduling parameters")
    if any(h not in state.capacity for h in state.placement.values()):
        raise ValueError("Unknown placement host")
    if any(c <= 0 or r <= 0 for c, r in state.capacity.values()):
        raise ValueError("Host capacities must be positive")
    if move_limit is None:
        move_limit = max(1, int(len(forecast) * 0.05))
    if move_limit < 0:
        raise ValueError("move_limit must be nonnegative")

    placement = dict(state.placement)
    loads = _loads(placement, forecast, state.capacity)
    moves, opened, unresolved = [], [], []
    source_hosts = sorted(state.capacity)
    for source in source_hosts:
        capacity = state.capacity[source]
        if not any(loads[source][h][r] > threshold * capacity[r]
                   for h in range(horizon) for r in range(2)):
            continue
        containers = [c for c, host in placement.items() if host == source]
        containers.sort(key=lambda c: (-max(
            forecast[c][h][r] / capacity[r] for h in range(horizon) for r in range(2)), c))
        for container in containers:
            if len(moves) >= move_limit:
                break
            if all(loads[source][h][r] <= threshold * capacity[r]
                   for h in range(horizon) for r in range(2)):
                break
            if minute - state.last_move.get(container, -10**9) < cooldown:
                continue
            candidates = []
            for host, cap in state.capacity.items():
                if host == source or state.ready_at.get(host, 0) > minute:
                    continue
                if all(loads[host][h][r] + forecast[container][h][r] <= cap[r]
                       for h in range(horizon) for r in range(2)):
                    # Best fit over the whole forecast window.
                    slack = sum((cap[r] - loads[host][h][r] -
                                 forecast[container][h][r]) / cap[r]
                                for h in range(horizon) for r in range(2))
                    candidates.append((slack, host))
            if candidates:
                target = min(candidates)[1]
            else:
                template = capacity
                if not all(forecast[container][h][r] <= template[r]
                           for h in range(horizon) for r in range(2)):
                    unresolved.append(container)
                    continue
                if any(ready > minute for ready in state.ready_at.values()):
                    unresolved.append(container)
                    break
                while f"new-{state.next_host}" in state.capacity:
                    state.next_host += 1
                target = f"new-{state.next_host}"
                state.next_host += 1
                state.capacity[target] = template
                state.ready_at[target] = minute + boot_minutes
                loads[target] = [[0.0, 0.0] for _ in range(horizon)]
                opened.append(target)
                if boot_minutes:
                    unresolved.append(container)
                    break
            placement[container] = target
            state.last_move[container] = minute
            moves.append((container, source, target))
            for h in range(horizon):
                for r in range(2):
                    loads[source][h][r] -= forecast[container][h][r]
                    loads[target][h][r] += forecast[container][h][r]
    state.placement = placement
    return Decision(dict(placement), moves, opened, unresolved)
