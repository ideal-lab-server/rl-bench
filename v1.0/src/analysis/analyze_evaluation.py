import json
from pathlib import Path
import numpy as np
import yaml
from utilities.results_utils import save_json


def load_json(path):
    with path.open() as file:
        return json.load(file)


def analyze_evaluation():
    benchmark_root = Path(__file__).resolve().parent.parent.parent

    config_path = benchmark_root / "configs" / "benchmark.yaml"

    with config_path.open() as file:
        config = yaml.safe_load(file)

    expected_seeds = config["training"]["seeds"]

    results_dir = benchmark_root / "results" / "reference"
    output_path = (
        benchmark_root
        / "results"
        / "summaries"
        / "evaluation_dynamics.json"
    )

    curve_files = sorted(results_dir.glob("*_curve.json"))

    if not curve_files:
        raise FileNotFoundError(
            f"No learning-curve files found in {results_dir}"
        )

    curves = {}

    for path in curve_files:
        curve = load_json(path)

        if not curve:
            raise ValueError(
                f"Learning curve is empty: {path.name}"
            )

        first = curve[0]

        algorithm = first["algorithm"]
        environment = first["environment"]
        training_seed = first["training_seed"]

        key = (algorithm, environment)

        if key not in curves:
            curves[key] = {}

        if training_seed in curves[key]:
            raise ValueError(
                f"Duplicate training seed {training_seed} "
                f"for {algorithm} + {environment}"
            )

        curves[key][training_seed] = curve

    analyses = []

    for (algorithm, environment), seed_curves in sorted(curves.items()):

        if len(seed_curves) != expected_seeds:
            raise ValueError(
                f"{algorithm} + {environment}: expected {expected_seeds} training "
                f"seeds, found {len(seed_curves)}"
            )

        seeds = sorted(seed_curves)

        reference_curve = seed_curves[seeds[0]]

        reference_timesteps = [
            entry["timesteps"]
            for entry in reference_curve
        ]

        for seed in seeds[1:]:
            timesteps = [
                entry["timesteps"]
                for entry in seed_curves[seed]
            ]

            if timesteps != reference_timesteps:
                raise ValueError(
                    f"Inconsistent checkpoints for "
                    f"{algorithm} + {environment}, "
                    f"seed {seed}"
                )

        checkpoints = []

        for checkpoint_index, timestep in enumerate(
            reference_timesteps
        ):
            seed_mean_returns = []

            for seed in seeds:
                evaluation = seed_curves[seed][checkpoint_index]

                if evaluation["timesteps"] != timestep:
                    raise ValueError(
                        f"Inconsistent timestep for "
                        f"{algorithm} + {environment}, "
                        f"seed {seed}"
                    )

                seed_mean_returns.append(
                    evaluation["mean_return"]
                )

            seed_mean_returns = np.asarray(
                seed_mean_returns,
                dtype=float,
            )

            checkpoints.append(
                {
                    "timesteps": timestep,
                    "training_seeds": seeds,
                    "seed_mean_returns": (
                        seed_mean_returns.tolist()
                    ),
                    "mean_return": float(
                        seed_mean_returns.mean()
                    ),
                    "std_return": float(
                        seed_mean_returns.std()
                    ),
                }
            )

        analyses.append(
            {
                "algorithm": algorithm,
                "environment": environment,
                "training_seed_count": len(seeds),
                "checkpoints": checkpoints,
            }
        )

    result = {
        "benchmark": "RL Benchmark",
        "version": config["benchmark"]["version"],
        "analysis": "evaluation_dynamics",
        "description": (
            "Evaluation dynamics aggregated across independent "
            "training seeds."
        ),
        "results": analyses,
    }

    save_json(result, output_path)

    print(f"Algorithm/environment pairs: {len(analyses)}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    analyze_evaluation()
