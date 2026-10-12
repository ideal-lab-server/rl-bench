import csv
from pathlib import Path
import yaml
from utilities.results_utils import save_json


def load_progress(path):
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames


def convert_value(value):
    if value == "":
        return None

    try:
        number = float(value)
    except ValueError:
        return value

    return int(number) if number.is_integer() else number


def analyze_training_logs():
    benchmark_root = Path(__file__).resolve().parent.parent.parent
    logs_dir = benchmark_root / "logs"
    output_path = benchmark_root / "results" / "summaries" / "training_dynamics.json"

    config_path = benchmark_root / "configs" / "benchmark.yaml"

    with config_path.open() as file:
        config = yaml.safe_load(file)

    log_files = sorted(logs_dir.glob("*/progress.csv"))

    if not log_files:
        raise FileNotFoundError(f"No progress.csv files found in {logs_dir}")

    results = []

    for path in log_files:
        rows, fields = load_progress(path)

        if not rows:
            raise ValueError(f"Training log is empty: {path}")

        required_fields = {
            "time/total_timesteps",
            "rollout/ep_rew_mean",
            "rollout/ep_len_mean",
        }

        missing = required_fields - set(fields)
        if missing:
            raise ValueError(f"{path}: missing fields: {sorted(missing)}")

        run_name = path.parent.name
        prefix, seed_part = run_name.rsplit("_seed", 1)
        algorithm, environment = prefix.split("_", 1)
        seed = int(seed_part)

        curve = []

        optional_fields = {
            "actor_loss": "train/actor_loss",
            "critic_loss": "train/critic_loss",
            "elapsed_seconds": "time/time_elapsed",
            "fps": "time/fps",
            "episodes": "time/episodes",
            "n_updates": "train/n_updates",
            "learning_rate": "train/learning_rate",
        }

        for row in rows:
            entry = {
                "timesteps": convert_value(row["time/total_timesteps"]),
                "episode_reward_mean": convert_value(row["rollout/ep_rew_mean"]),
                "episode_length_mean": convert_value(row["rollout/ep_len_mean"]),
            }

            for output_name, csv_name in optional_fields.items():
                if csv_name in fields:
                    entry[output_name] = convert_value(row[csv_name])

            curve.append(entry)

        results.append({
            "algorithm": algorithm,
            "environment": environment,
            "training_seed": seed,
            "run_name": run_name,
            "source": str(path),
            "fields": fields,
            "curve": curve,
        })

    results.sort(
        key=lambda item: (
            item["algorithm"],
            item["environment"],
            item["training_seed"],
        )
    )

    output = {
        "benchmark": "RL Benchmark",
        "version": config["benchmark"]["version"],
        "analysis": "training_dynamics",
        "description": (
            "Training dynamics extracted from Stable-Baselines3 "
            "progress.csv logs."
        ),
        "runs": results,
    }

    save_json(output, output_path)

    print(f"Training log files: {len(log_files)}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    analyze_training_logs()
