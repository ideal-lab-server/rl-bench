import json
from pathlib import Path

from utilities.results_utils import save_json


def load_json(path):
    with path.open() as file:
        return json.load(file)


def main():
    root = Path(__file__).resolve().parent.parent.parent

    input_path = (
        root
        / "results"
        / "summaries"
        / "evaluation_dynamics.json"
    )

    output_path = (
        root
        / "results"
        / "summaries"
        / "final_performance.json"
    )

    data = load_json(input_path)

    results = []

    for item in data["results"]:
        checkpoints = item["checkpoints"]

        if not checkpoints:
            raise ValueError(
                f"No checkpoints found for "
                f"{item['algorithm']} + {item['environment']}"
            )

        final = checkpoints[-1]

        results.append(
            {
                "algorithm": item["algorithm"],
                "environment": item["environment"],
                "timesteps": final["timesteps"],
                "training_seed_count": item[
                    "training_seed_count"
                ],
                "training_seeds": final[
                    "training_seeds"
                ],
                "seed_mean_returns": final[
                    "seed_mean_returns"
                ],
                "mean_return": final[
                    "mean_return"
                ],
                "std_return": final[
                    "std_return"
                ],
            }
        )

    output = {
        "benchmark": data["benchmark"],
        "version": data["version"],
        "analysis": "final_performance",
        "description": (
            "Final deterministic evaluation performance "
            "at the end of each environment-specific training budget, aggregated "
            "across independent training seeds."
        ),
        "results": results,
    }

    save_json(output, output_path)

    print(f"Algorithm/environment pairs: {len(results)}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
