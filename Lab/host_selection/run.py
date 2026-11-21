"""Run B0 (last observation) and B1 (provided forecast) on the same trace.

JSON format: capacity {host:[cpu,ram]}, placement {container:host},
actual [{container:[cpu,ram]}, ...], forecast [{container:[[cpu,ram], ...]}, ...].
At minute t, actual[t] is hidden from the scheduler. B0 uses actual[t-1].
"""
import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from statistics import median
from scheduler import State, schedule


def run(data, mode):
    actual = data["actual"]
    forecasts = data["forecast"]
    if len(actual) != len(forecasts) or len(actual) < 2:
        raise ValueError("Need aligned actual and forecast with at least 2 minutes")
    state = State({k: tuple(v) for k, v in data["capacity"].items()},
                  dict(data["placement"]))
    over, short_cpu, short_ram, host_steps, active_steps, moves = 0, 0.0, 0.0, 0, 0, 0
    durations = []
    for t in range(1, len(actual)):
        if mode == "B0":
            horizon = len(next(iter(forecasts[t].values())))
            predicted = {c: [tuple(actual[t-1][c])] * horizon for c in state.placement}
        else:
            predicted = {c: [tuple(v) for v in forecasts[t][c]]
                         for c in state.placement}
        started = time.perf_counter()
        result = schedule(state, predicted, t,
                          boot_minutes=data.get("boot_minutes", 1),
                          move_limit=data.get("move_limit"))
        durations.append((time.perf_counter() - started) * 1000)
        moves += len(result.moves)
        loads = {h: [0.0, 0.0] for h in state.capacity}
        for c, host in result.placement.items():
            for r in range(2):
                loads[host][r] += actual[t][c][r]
        for host, values in loads.items():
            active_steps += 1
            if state.ready_at.get(host, 0) > t:
                continue
            cap = state.capacity[host]
            host_steps += 1
            if any(values[r] > cap[r] for r in range(2)):
                over += 1
            short_cpu += max(0, values[0] - cap[0])
            short_ram += max(0, values[1] - cap[1])
    bucket_seconds = data.get("bucket_seconds", 60)
    ordered = sorted(durations)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return {"mode": mode, "overloaded_host_steps": over,
            "observed_host_steps": host_steps,
            "overload_rate": over / host_steps if host_steps else None,
            "cpu_shortfall": short_cpu, "ram_shortfall": short_ram,
            "active_host_hours": active_steps * bucket_seconds / 3600,
            "moves": moves,
            "moves_per_100_container_hours":
            moves * 100 * 3600 / ((len(actual)-1) * len(state.placement) * bucket_seconds),
            "decision_ms_median": median(durations), "decision_ms_p95": p95}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    result = [run(deepcopy(data), "B0"), run(deepcopy(data), "B1")]
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
