"""Evaluate a prespecified margin on identical RNN comparison files."""
import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

from margin import compare


def experiment(comparison_dir, q=0.95):
    paths = sorted(comparison_dir.glob("comparison_seed_*.json"))
    if not paths:
        raise ValueError("No comparison_seed_*.json files found")
    rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        result = compare(data, q)
        rows.append({"seed": data.get("metadata", {}).get("seed"),
                     "source_file": str(path), **result})
    aggregate = {}
    for mode in ("B1", "B2"):
        aggregate[mode] = {}
        for key in ("overload_rate", "cpu_shortfall", "ram_shortfall",
                    "active_host_hours", "moves_per_100_container_hours"):
            values = [row[mode][key] for row in rows]
            if all(value is not None for value in values):
                aggregate[mode][key] = {
                    "mean": mean(values), "std_population": pstdev(values),
                    "min": min(values), "max": max(values)}
    return {"quantile_prespecified": q, "per_seed": rows,
            "aggregate": aggregate}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--q", type=float, default=0.95)
    args = parser.parse_args()
    result = experiment(args.comparison_dir, args.q)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))
