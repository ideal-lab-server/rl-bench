import json
from pathlib import Path
import yaml
from utilities.results_utils import save_json


def load_json(path):
    with path.open() as file:
        return json.load(file)


def main():
    root = Path(__file__).resolve().parent.parent.parent

    config_path = root / "configs" / "benchmark.yaml"

    with config_path.open() as file:
        config = yaml.safe_load(file)

    results_dir = root / "results" / "reference"

    output_path = (
        root
        / "results"
        / "summaries"
        / "training_time.json"
    )

    files = sorted(results_dir.glob("*_train.json"))

    if not files:
        raise FileNotFoundError(
            f"No training results found in {results_dir}"
        )

    runs = []

    for path in files:
        data = load_json(path)

        required = {
            "algorithm",
            "environment",
            "seed",
            "timesteps",
            "device",
            "training_time_seconds",
        }

        missing = required - data.keys()

        if missing:
            raise ValueError(
                f"{path.name}: missing fields: "
                f"{sorted(missing)}"
            )

        runs.append(
            {
                "algorithm": data["algorithm"],
                "environment": data["environment"],
                "seed": data["seed"],
                "timesteps": data["timesteps"],
                "device": data["device"],
                "training_time_seconds": data[
                    "training_time_seconds"
                ],
            }
        )

    groups = {}

    for run in runs:
        key = (
            run["algorithm"],
            run["environment"],
        )

        groups.setdefault(key, []).append(run)

    summaries = []

    for (algorithm, environment), group in sorted(
        groups.items()
    ):
        times = [
            run["training_time_seconds"]
            for run in group
        ]

        mean_time = sum(times) / len(times)

        variance = sum(
            (time - mean_time) ** 2
            for time in times
        ) / len(times)

        summaries.append(
            {
                "algorithm": algorithm,
                "environment": environment,
                "training_seed_count": len(group),
                "training_seeds": sorted(
                    run["seed"] for run in group
                ),
                "timesteps": group[0]["timesteps"],
                "device": group[0]["device"],
                "training_time_seconds": times,
                "mean_training_time_seconds": mean_time,
                "std_training_time_seconds": variance ** 0.5,
            }
        )

    output = {
        "benchmark": "RL Benchmark",
        "version": config["benchmark"]["version"],
        "analysis": "training_time",
        "description": (
            "Wall-clock training time aggregated across "
            "independent training seeds."
        ),
        "run_count": len(runs),
        "results": summaries,
    }

    save_json(output, output_path)

    print(f"Training runs: {len(runs)}")
    print(
        f"Algorithm/environment pairs: {len(summaries)}"
    )
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
