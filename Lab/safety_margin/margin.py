"""One-sided residual-quantile safety margin for CPU/RAM forecasts."""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "host_selection"))
from run import run


def calibrate(errors, q):
    values = np.asarray(errors, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2 or not len(values):
        raise ValueError("Expected calibration errors [samples,horizon,CPU/RAM]")
    if not 0 < q < 1:
        raise ValueError("q must be between zero and one")
    return np.maximum(np.quantile(values, q, axis=0), 0)


def apply_margin(comparison, q):
    margin = calibrate(comparison["calibration_errors"], q)
    result = deepcopy(comparison)
    for step in result["forecast"]:
        for container, forecast in step.items():
            arr = np.asarray(forecast, dtype=float)
            if arr.shape != margin.shape:
                raise ValueError("Forecast horizon differs from calibration")
            step[container] = (arr + margin).tolist()
    result["margin"] = {"quantile": q, "cpu_ram_by_horizon": margin.tolist(),
                        "calibration_samples": len(comparison["calibration_errors"])}
    return result


def empirical_coverage(comparison):
    """Share of observed demands below each forecast bound, by horizon/resource."""
    forecast = comparison["forecast"]
    actual = comparison["actual"]
    horizon = len(next(iter(forecast[0].values())))
    hits = np.zeros((horizon, 2), dtype=float)
    counts = np.zeros((horizon, 2), dtype=int)
    for t in range(1, len(forecast)):
        step = forecast[t]
        for h in range(horizon):
            if t + h >= len(actual):
                continue
            for container, prediction in step.items():
                observed = actual[t+h].get(container)
                if observed is None:
                    continue
                hits[h] += (np.asarray(observed) <= np.asarray(prediction[h]))
                counts[h] += 1
    return np.divide(hits, counts, out=np.zeros_like(hits),
                     where=counts > 0).tolist()


def compare(comparison, quantile=0.95):
    """Primary test evaluation with q fixed before observing test metrics."""
    baseline = run(comparison, "B1")
    adjusted = apply_margin(comparison, quantile)
    metrics = run(adjusted, "B1")
    return {"B1": baseline, "B2": {**metrics, "mode": "B2"},
            "margin": adjusted["margin"],
            "empirical_test_coverage_B1": empirical_coverage(comparison),
            "empirical_test_coverage_B2": empirical_coverage(adjusted),
            "note": "q was prespecified before test. Coverage is empirical, not a guarantee."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--q", type=float, default=0.95,
                        help="Prespecified quantile; do not choose using test results")
    args = parser.parse_args()
    data = json.loads(args.comparison.read_text(encoding="utf-8"))
    result = compare(data, args.q)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
