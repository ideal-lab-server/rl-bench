import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path):
    with path.open() as file:
        return json.load(file)


def main():
    root = Path(__file__).resolve().parent.parent.parent
    summaries = root / "results" / "summaries"
    plots = root / "plots"

    plots.mkdir(parents=True, exist_ok=True)

    training = load_json(
        summaries / "training_dynamics.json"
    )
    evaluation = load_json(
        summaries / "evaluation_dynamics.json"
    )

    # ------------------------------------------------------------------
    # Training curves
    # ------------------------------------------------------------------

    training_groups = {}

    for run in training["runs"]:
        key = (run["algorithm"], run["environment"])
        training_groups.setdefault(key, []).append(run)

    for (algorithm, environment), runs in sorted(
        training_groups.items()
    ):
        plt.figure(figsize=(8, 5))

        for run in sorted(
            runs,
            key=lambda x: x["training_seed"],
        ):
            points = [
                entry
                for entry in run["curve"]
                if entry["episode_reward_mean"] is not None
            ]

            x = [
                entry["timesteps"]
                for entry in points
            ]
            y = [
                entry["episode_reward_mean"]
                for entry in points
            ]

            plt.plot(
                x,
                y,
                alpha=0.45,
                linewidth=1,
                label=f"Seed {run['training_seed']}",
            )

        # Mean across training seeds at each logged timestep.
        curves = {
            run["training_seed"]: {
                entry["timesteps"]: entry["episode_reward_mean"]
                for entry in run["curve"]
                if entry["episode_reward_mean"] is not None
            }
            for run in runs
        }

        all_steps = sorted(
            set.union(
                *(set(curve) for curve in curves.values())
            )
        )

        common_steps = []
        mean_values = []
        std_values = []

        for step in all_steps:
            values = [
                curves[seed][step]
                for seed in curves
                if step in curves[seed]
            ]

            if not values:
                continue

            common_steps.append(step)
            mean_values.append(np.mean(values))
            std_values.append(np.std(values))

        mean_values = np.asarray(mean_values)
        std_values = np.asarray(std_values)

        plt.plot(
            common_steps,
            mean_values,
            linewidth=2,
            label="Mean",
        )

        plt.fill_between(
            common_steps,
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.15,
        )

        plt.xlabel("Training steps")
        plt.ylabel("Mean episode return")
        plt.title(
            f"Training Curve: {algorithm} + {environment}"
        )
        plt.legend()
        plt.tight_layout()

        filename = (
            f"training_{algorithm}_{environment}.png"
        )

        plt.savefig(
            plots / filename,
            dpi=200,
        )
        plt.close()

    # ------------------------------------------------------------------
    # Evaluation curves
    # ------------------------------------------------------------------

    for result in evaluation["results"]:
        algorithm = result["algorithm"]
        environment = result["environment"]

        checkpoints = result["checkpoints"]

        x = [
            checkpoint["timesteps"]
            for checkpoint in checkpoints
        ]

        mean = np.asarray(
            [
                checkpoint["mean_return"]
                for checkpoint in checkpoints
            ]
        )

        std = np.asarray(
            [
                checkpoint["std_return"]
                for checkpoint in checkpoints
            ]
        )

        plt.figure(figsize=(8, 5))

        plt.plot(
            x,
            mean,
            linewidth=2,
            label="Mean across seeds",
        )

        plt.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.2,
            label="±1 SD",
        )

        plt.xlabel("Training steps")
        plt.ylabel("Mean episode return")
        plt.title(
            f"Evaluation Curve: {algorithm} + {environment}"
        )
        plt.legend()
        plt.tight_layout()

        filename = (
            f"evaluation_{algorithm}_{environment}.png"
        )

        plt.savefig(
            plots / filename,
            dpi=200,
        )
        plt.close()

    print("Training plots:", len(training_groups))
    print("Evaluation plots:", len(evaluation["results"]))
    print(f"Plots saved to: {plots}")


if __name__ == "__main__":
    main()
