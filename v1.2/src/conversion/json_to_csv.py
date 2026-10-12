import argparse
import json
from pathlib import Path

import pandas as pd


def convert_final_performance(data):
    """Convert final-performance summary to one row per combination."""
    rows = []

    for result in data["results"]:
        row = {
            "algorithm": result["algorithm"],
            "environment": result["environment"],
            "timesteps": result["timesteps"],
            "training_seed_count": result["training_seed_count"],
        }

        for seed, value in zip(
            result["training_seeds"],
            result["seed_mean_returns"],
        ):
            row[f"seed_{seed}_return"] = value

        row["mean_return"] = result["mean_return"]
        row["std_return"] = result["std_return"]

        rows.append(row)

    return pd.DataFrame(rows)


def convert_training_time(data):
    """Convert training-time summary to one row per combination."""
    rows = []

    for result in data["results"]:
        row = {
            "algorithm": result["algorithm"],
            "environment": result["environment"],
            "training_seed_count": result["training_seed_count"],
            "timesteps": result["timesteps"],
            "device": result["device"],
        }

        for seed, value in zip(
            result["training_seeds"],
            result["training_time_seconds"],
        ):
            row[f"seed_{seed}_time_seconds"] = value

        row["mean_training_time_seconds"] = result[
            "mean_training_time_seconds"
        ]
        row["std_training_time_seconds"] = result[
            "std_training_time_seconds"
        ]

        rows.append(row)

    return pd.DataFrame(rows)


def convert_evaluation_dynamics(data):
    """Convert evaluation dynamics to one row per checkpoint."""
    rows = []

    for result in data["results"]:
        for checkpoint in result["checkpoints"]:
            rows.append(
                {
                    "algorithm": result["algorithm"],
                    "environment": result["environment"],
                    "timesteps": checkpoint["timesteps"],
                    "training_seed_count": len(
                        checkpoint["training_seeds"]
                    ),
                    "mean_return": checkpoint["mean_return"],
                    "std_return": checkpoint["std_return"],
                }
            )

    return pd.DataFrame(rows)


def convert_training_dynamics(data):
    """Convert training dynamics to one row per recorded checkpoint."""
    rows = []

    for run in data["runs"]:
        for checkpoint in run["curve"]:
            row = {
                "algorithm": run["algorithm"],
                "environment": run["environment"],
                "training_seed": run["training_seed"],
                "run_name": run["run_name"],
            }
            row.update(checkpoint)
            rows.append(row)

    return pd.DataFrame(rows)


def convert_json(data):
    """Convert a benchmark summary JSON object to a DataFrame."""
    analysis = data.get("analysis")

    converters = {
        "final_performance": convert_final_performance,
        "training_time": convert_training_time,
        "evaluation_dynamics": convert_evaluation_dynamics,
        "training_dynamics": convert_training_dynamics,
    }

    if analysis not in converters:
        raise ValueError(
            f"Unsupported benchmark analysis type: {analysis!r}"
        )

    return converters[analysis](data)


def main():
    parser = argparse.ArgumentParser(
        description="Convert RL benchmark JSON summaries to CSV."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input JSON summary.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output CSV file.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open() as file:
        data = json.load(file)

    dataframe = convert_json(data)

    if dataframe.empty:
        raise ValueError("Converted data contains no rows.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)

    print(f"Converted: {input_path}")
    print(f"Written:   {output_path}")
    print(f"Rows:      {len(dataframe)}")
    print(f"Columns:   {len(dataframe.columns)}")


if __name__ == "__main__":
    main()
