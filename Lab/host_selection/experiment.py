"""Run three-seed B0/B1 experiment on a prepared trace.

Writes individual comparisons and one aggregate JSON report. This command
does not download data or infer whether an observed difference is significant.
"""
import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

from train_rnn import build
from run import run


def experiment(prepared, output_dir, seeds=(11, 22, 33),
               history=6, horizon=1, hidden=16, epochs=30):
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed = []
    for seed in seeds:
        comparison = build(prepared, history=history, horizon=horizon,
                           hidden=hidden, epochs=epochs, seed=seed)
        (output_dir / f"comparison_seed_{seed}.json").write_text(
            json.dumps(comparison, ensure_ascii=False), encoding="utf-8")
        b0 = run(comparison, "B0")
        b1 = run(comparison, "B1")
        per_seed.append({"seed": seed, "metadata": comparison["metadata"],
                         "B0": b0, "B1": b1})
    metrics = ("overload_rate", "cpu_shortfall", "ram_shortfall",
               "active_host_hours", "moves_per_100_container_hours",
               "decision_ms_median", "decision_ms_p95")
    aggregate = {}
    for mode in ("B0", "B1"):
        aggregate[mode] = {}
        for metric in metrics:
            values = [row[mode][metric] for row in per_seed]
            if all(v is not None for v in values):
                aggregate[mode][metric] = {
                    "mean": mean(values), "std_population": pstdev(values),
                    "min": min(values), "max": max(values)}
    report = {"source": prepared.get("source"),
              "resource_units": prepared.get("resource_units"),
              "bucket_seconds": prepared["bucket_seconds"],
              "seeds": list(seeds), "per_seed": per_seed,
              "aggregate": aggregate,
              "interpretation": "Simulation results; no SLA, energy or cost claim."}
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--history", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    prepared = json.loads(args.prepared.read_text(encoding="utf-8"))
    report = experiment(prepared, args.output_dir, args.seeds,
                        args.history, args.horizon, args.hidden, args.epochs)
    print(json.dumps(report["aggregate"], indent=2))
