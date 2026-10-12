import argparse
from pathlib import Path

from utilities.results_utils import load_json, save_json


def main():
    parser = argparse.ArgumentParser(
        description="Extract the final evaluation from a learning curve."
    )

    parser.add_argument("--curve", required=True)
    parser.add_argument("--result", required=True)

    args = parser.parse_args()

    curve = load_json(args.curve)

    if not curve:
        raise ValueError("Learning curve is empty.")

    final = curve[-1]

    required = {
        "algorithm",
        "environment",
        "training_seed",
        "timesteps",
        "episodes",
        "deterministic",
        "evaluation_seeds",
        "episode_returns",
        "mean_return",
        "std_return",
    }

    missing = required - final.keys()

    if missing:
        raise ValueError(
            f"Final evaluation is missing fields: {sorted(missing)}"
        )

    result = {
        "algorithm": final["algorithm"],
        "environment": final["environment"],
        "episodes": final["episodes"],
        "deterministic": final["deterministic"],
        "device": "cpu",
        "training_seed": final["training_seed"],
        "evaluation_seeds": final["evaluation_seeds"],
        "episode_returns": final["episode_returns"],
        "mean_return": final["mean_return"],
        "std_return": final["std_return"],
    }

    save_json(result, Path(args.result))

    print(f"Final evaluation: {final['timesteps']} steps")
    print(f"Mean return:      {final['mean_return']:.2f}")
    print(f"Std. deviation:   {final['std_return']:.2f}")
    print(f"Result saved to:  {args.result}")


if __name__ == "__main__":
    main()
