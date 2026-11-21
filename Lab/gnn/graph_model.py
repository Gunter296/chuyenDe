"""Trainable one-layer graph convolution for multi-container load forecasting.

Edges are fixed using CPU correlation from the training interval only.
Node features are each container's recent CPU/RAM history. A learned
message layer mixes neighboring histories before predicting future load.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "host_selection"))
from run import run


def graph(actual, containers, train_end, neighbors=2):
    n = len(containers)
    matrix = np.zeros((n, n))
    for i, a in enumerate(containers):
        for j in range(i + 1, n):
            b = containers[j]
            pairs = [(row[a][0], row[b][0]) for row in actual[:train_end]
                     if a in row and b in row]
            if len(pairs) >= 3:
                x, y = np.asarray(pairs).T
                if np.std(x) > 0 and np.std(y) > 0:
                    matrix[i, j] = matrix[j, i] = max(0, np.corrcoef(x, y)[0, 1])
    adjacency = np.zeros_like(matrix)
    for i in range(n):
        for j in np.argsort(-matrix[i], kind="stable")[:min(neighbors, n-1)]:
            if i != j and matrix[i, j] > 0:
                adjacency[i, j] = matrix[i, j]
    rows = adjacency.sum(axis=1, keepdims=True)
    return adjacency / np.maximum(rows, 1), matrix


def examples(actual, containers, lo, hi, history, horizon, scale):
    for t in range(max(lo, history), hi-horizon+1):
        if any(any(c not in actual[k] for c in containers)
               for k in range(t-history, t+horizon)):
            continue
        x = np.asarray([[actual[k][c] for k in range(t-history, t)]
                        for c in containers]) / scale
        y = np.asarray([[actual[k][c] for k in range(t, t+horizon)]
                        for c in containers]) / scale
        yield t, x.reshape(len(containers), -1), y.reshape(len(containers), -1)


class GraphNet:
    def __init__(self, inputs, outputs, hidden, seed):
        rng = np.random.default_rng(seed)
        self.ws = rng.normal(0, 0.08, (inputs, hidden))
        self.wn = rng.normal(0, 0.08, (inputs, hidden))
        self.b = np.zeros(hidden)
        self.v = rng.normal(0, 0.08, (hidden, outputs))
        self.c = np.zeros(outputs)

    def predict(self, x, adjacency):
        neighbor = adjacency @ x
        hidden = np.tanh(x @ self.ws + neighbor @ self.wn + self.b)
        return hidden @ self.v + self.c, hidden, neighbor

    def step(self, x, y, adjacency, lr):
        prediction, hidden, neighbor = self.predict(x, adjacency)
        delta = 2 * (prediction - y) / prediction.size
        gv = hidden.T @ delta
        gc = delta.sum(axis=0)
        dh = (delta @ self.v.T) * (1 - hidden ** 2)
        gws = x.T @ dh
        gwn = neighbor.T @ dh
        gb = dh.sum(axis=0)
        gradients = [gws, gwn, gb, gv, gc]
        norm = np.sqrt(sum(np.sum(g*g) for g in gradients))
        factor = min(1, 1 / (norm + 1e-12))
        for p, g in zip((self.ws, self.wn, self.b, self.v, self.c), gradients):
            p -= lr * factor * g


def build(prepared, history=6, horizon=1, hidden=16, epochs=30,
          seed=11, neighbors=2, capacity=(100., 100.),
          graph_enabled=True):
    actual, containers = prepared["actual"], prepared["containers"]
    n = len(actual)
    train_end, val_end = int(n*0.6), int(n*0.8)
    selection_end = train_end + (val_end-train_end)//2
    if train_end < history+horizon:
        raise ValueError("Trace too short")
    scale_values = np.asarray([row[c] for row in actual[:train_end]
                               for c in containers if c in row])
    scale = np.maximum(np.quantile(scale_values, 0.99, axis=0), 1e-6)
    adjacency, correlation = graph(actual, containers, train_end, neighbors)
    if not graph_enabled:
        adjacency = np.zeros_like(adjacency)
    train = list(examples(actual, containers, 0, train_end, history, horizon, scale))
    validation = list(examples(actual, containers, train_end, selection_end,
                               history, horizon, scale))
    calibration = list(examples(actual, containers, selection_end, val_end,
                                history, horizon, scale))
    if not train or not validation or not calibration:
        raise ValueError("No complete train/validation/calibration windows")
    model = GraphNet(history*2, horizon*2, hidden, seed)
    rng = np.random.default_rng(seed)
    best, stale, params = float("inf"), 0, None
    training_started = time.perf_counter()
    epochs_ran = 0
    for epoch in range(epochs):
        epochs_ran = epoch + 1
        for i in rng.permutation(len(train)):
            _, x, y = train[i]
            model.step(x, y, adjacency, 0.01)
        loss = np.mean([np.mean((model.predict(x, adjacency)[0]-y)**2)
                        for _, x, y in validation])
        if loss < best:
            best, stale = float(loss), 0
            params = [p.copy() for p in
                      (model.ws, model.wn, model.b, model.v, model.c)]
        else:
            stale += 1
            if stale >= 5:
                break
    training_seconds = time.perf_counter() - training_started
    model.ws, model.wn, model.b, model.v, model.c = params
    residuals = []
    for _, x, y in calibration:
        estimate = model.predict(x, adjacency)[0].reshape(len(containers), horizon, 2)
        residuals.extend(((y.reshape(len(containers), horizon, 2)-estimate)
                          * scale).tolist())
    test = list(examples(actual, containers, val_end, n, history, horizon, scale))
    runs, start = [], 0
    for i in range(1, len(test)+1):
        if i == len(test) or test[i][0] != test[i-1][0]+1:
            runs.append(test[start:i])
            start = i
    test = max(runs, key=len) if runs else []
    if len(test) < 2:
        raise ValueError("Fewer than two consecutive complete test steps")
    truth, forecasts, errors = [], [], []
    inference_started = time.perf_counter()
    for t, x, y in test:
        estimate = np.maximum(model.predict(x, adjacency)[0]
                              .reshape(len(containers), horizon, 2)*scale, 0)
        forecast = {c: estimate[i].tolist() for i, c in enumerate(containers)}
        forecasts.append(forecast)
        truth.append({c: actual[t][c] for c in containers})
        errors.append(estimate - y.reshape(len(containers), horizon, 2)*scale)
    error = np.asarray(errors)
    inference_ms_per_step = (time.perf_counter() - inference_started) * 1000 / len(test)
    count = max(2, len(containers)//5)
    return {"capacity": {f"h{i}": list(capacity) for i in range(count)},
            "placement": {c: f"h{i%count}" for i, c in enumerate(containers)},
            "actual": truth, "forecast": forecasts,
            "calibration_errors": residuals,
            "bucket_seconds": prepared["bucket_seconds"], "boot_minutes": 1,
            "metadata": {"model": "GraphNet" if graph_enabled else "NoGraph",
                         "seed": seed, "train_end": train_end,
                         "selection_end": selection_end, "val_end": val_end,
                         "edge_count": int(np.count_nonzero(adjacency)),
                         "training_seconds": training_seconds,
                         "epochs_ran": epochs_ran,
                         "inference_ms_per_step": inference_ms_per_step,
                         "train_windows": len(train),
                         "validation_windows": len(validation),
                         "calibration_samples": len(residuals),
                         "test_steps": len(test),
                         "test_mae_by_horizon_cpu_ram":
                         np.mean(np.abs(error), axis=(0,1)).tolist(),
                         "test_rmse_by_horizon_cpu_ram":
                         np.sqrt(np.mean(error**2, axis=(0,1))).tolist()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    prepared = json.loads(args.prepared.read_text(encoding="utf-8"))
    comparison = build(prepared, seed=args.seed,
                       graph_enabled=not args.no_graph)
    args.output.write_text(json.dumps(comparison), encoding="utf-8")
    print(json.dumps({"forecast": comparison["metadata"],
                      "scheduler": run(comparison, "B1")}, indent=2))
