"""Train a small shared Elman RNN and build B0/B1 simulation input.

Uses only NumPy. Forecasts are generated from observations strictly before t.
Rows with missing history or targets are excluded, never filled with zero.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def windows(actual, containers, history, horizon, lo, hi):
    for t in range(max(lo, history), hi - horizon + 1):
        for c in containers:
            past = [actual[k].get(c) for k in range(t - history, t)]
            future = [actual[k].get(c) for k in range(t, t + horizon)]
            if all(x is not None for x in past + future):
                yield np.asarray(past, dtype=float), np.asarray(future, dtype=float)


class RNN:
    def __init__(self, hidden, horizon, seed):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.15, (2, hidden))
        self.u = rng.normal(0, 0.1, (hidden, hidden))
        self.b = np.zeros(hidden)
        self.v = rng.normal(0, 0.1, (hidden, 2 * horizon))
        self.c = np.zeros(2 * horizon)
        self.horizon = horizon

    def predict(self, x):
        states = [np.zeros(len(self.b))]
        for row in x:
            states.append(np.tanh(row @ self.w + states[-1] @ self.u + self.b))
        return states[-1] @ self.v + self.c, states

    def step(self, x, target, lr):
        y, states = self.predict(x)
        dy = 2 * (y - target.ravel()) / y.size
        grad_v = np.outer(states[-1], dy)
        grad_c = dy
        grad_w = np.zeros_like(self.w)
        grad_u = np.zeros_like(self.u)
        grad_b = np.zeros_like(self.b)
        dh = dy @ self.v.T
        for j in range(len(x) - 1, -1, -1):
            dz = dh * (1 - states[j + 1] ** 2)
            grad_w += np.outer(x[j], dz)
            grad_u += np.outer(states[j], dz)
            grad_b += dz
            dh = dz @ self.u.T
        gradients = [grad_w, grad_u, grad_b, grad_v, grad_c]
        norm = np.sqrt(sum(np.sum(g * g) for g in gradients))
        scale = min(1.0, 1.0 / (norm + 1e-12))
        for param, grad in zip((self.w, self.u, self.b, self.v, self.c), gradients):
            param -= lr * scale * grad
        return float(np.mean((y - target.ravel()) ** 2))


def build(prepared, history=6, horizon=1, hidden=16, epochs=15,
          seed=11, capacity=(100.0, 100.0)):
    actual = prepared["actual"]
    containers = prepared["containers"]
    n = len(actual)
    train_end, val_end = int(n * 0.6), int(n * 0.8)
    if train_end < history + horizon or val_end >= n - horizon:
        raise ValueError("Trace is too short for time split and windows")
    train = list(windows(actual, containers, history, horizon, 0, train_end))
    selection_end = train_end + (val_end - train_end) // 2
    val = list(windows(actual, containers, history, horizon, train_end, selection_end))
    calibration = list(windows(actual, containers, history, horizon,
                               selection_end, val_end))
    if not train or not val or not calibration:
        raise ValueError("No complete training/validation windows")
    values = np.concatenate([x for x, _ in train])
    scale = np.maximum(np.quantile(values, 0.99, axis=0), 1e-6)
    model = RNN(hidden, horizon, seed)
    rng = np.random.default_rng(seed)
    best = float("inf")
    best_params = None
    stale = 0
    for epoch in range(epochs):
        for index in rng.permutation(len(train)):
            x, y = train[index]
            model.step(x / scale, y / scale, 0.001)
        loss = float(np.mean([np.mean(
            (model.predict(x / scale)[0] - (y / scale).ravel()) ** 2)
            for x, y in val]))
        if loss < best:
            best, stale = loss, 0
            best_params = [p.copy() for p in
                           (model.w, model.u, model.b, model.v, model.c)]
        else:
            stale += 1
            if stale >= 5:
                break
    model.w, model.u, model.b, model.v, model.c = best_params
    calibration_errors = [
        (y - model.predict(x / scale)[0].reshape(horizon, 2) * scale).tolist()
        for x, y in calibration
    ]
    # Keep only test steps with complete histories and all present actual labels.
    test_actual, test_forecast, test_times = [], [], []
    for t in range(val_end, n - horizon + 1):
        if any(any(actual[k].get(c) is None for k in range(t-history, t+horizon))
               for c in containers):
            continue
        observation = {c: actual[t][c] for c in containers}
        prediction = {}
        for c in containers:
            x = np.asarray([actual[k][c] for k in range(t-history, t)])
            y = model.predict(x / scale)[0].reshape(horizon, 2) * scale
            prediction[c] = np.maximum(y, 0).tolist()
        test_actual.append(observation)
        test_forecast.append(prediction)
        test_times.append(t)
    # B0 must use the immediately preceding trace bucket, not a retained
    # sample separated by a missing-data gap.
    runs = []
    start = 0
    for i in range(1, len(test_times) + 1):
        if i == len(test_times) or test_times[i] != test_times[i-1] + 1:
            runs.append((start, i))
            start = i
    complete_test_steps = len(test_times)
    if runs:
        start, end = max(runs, key=lambda pair: pair[1] - pair[0])
        test_actual = test_actual[start:end]
        test_forecast = test_forecast[start:end]
        test_times = test_times[start:end]
    if len(test_actual) < 2:
        raise ValueError("Fewer than two complete test steps")
    errors = np.asarray([
        [np.asarray(test_forecast[i][c]) -
         np.asarray([actual[t + h][c] for h in range(horizon)])
         for c in containers]
        for i, t in enumerate(test_times)
    ])
    mae = np.mean(np.abs(errors), axis=(0, 1))
    rmse = np.sqrt(np.mean(errors ** 2, axis=(0, 1)))
    # Initial placement is deterministic and is shared by B0/B1.
    host_count = max(2, len(containers) // 5)
    hosts = {f"h{i}": list(capacity) for i in range(host_count)}
    placement = {c: f"h{i % host_count}" for i, c in enumerate(containers)}
    return {"capacity": hosts, "placement": placement, "actual": test_actual,
            "forecast": test_forecast, "boot_minutes": 1,
            "bucket_seconds": prepared["bucket_seconds"],
            "calibration_errors": calibration_errors,
            "metadata": {"train_end": train_end, "val_end": val_end,
                         "selection_end": selection_end,
                         "train_windows": len(train), "validation_windows": len(val),
                         "calibration_windows": len(calibration),
                         "validation_mse_normalized": best, "scale": scale.tolist(),
                         "test_mae_by_horizon_cpu_ram": mae.tolist(),
                         "test_rmse_by_horizon_cpu_ram": rmse.tolist(),
                         "seed": seed, "history": history, "horizon": horizon,
                         "test_steps": len(test_actual),
                         "discarded_noncontiguous_test_steps":
                         complete_test_steps - len(test_actual)}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--history", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()
    data = build(json.loads(args.prepared.read_text(encoding="utf-8")),
                 history=args.history, horizon=args.horizon, seed=args.seed)
    args.output.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(data["metadata"], indent=2))
