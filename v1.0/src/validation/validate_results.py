import json
from pathlib import Path

import yaml
from utilities.benchmark_utils import get_valid_combinations


def load_json(path):
    with path.open() as file:
        return json.load(file)


def load_config(path):
    with path.open() as file:
        return yaml.safe_load(file)


def main():
    benchmark_root = Path(__file__).resolve().parent.parent.parent
    config_path = benchmark_root / "configs" / "benchmark.yaml"
    results_dir = benchmark_root / "results" / "reference"

    config = load_config(config_path)

    environment_budgets = config["environment_budget"]
    seeds = config["training"]["seeds"]
    episodes = config["evaluation"]["episodes"]

    expected_seeds = set(range(seeds))
    expected_combinations = set(get_valid_combinations())

    train_files = sorted(results_dir.glob("*_train.json"))
    eval_files = sorted(results_dir.glob("*_eval.json"))

    expected_runs = len(expected_combinations) * len(expected_seeds)

    errors = []

    train_runs = {}
    eval_runs = {}

    # ------------------------------------------------------------
    # Check training results
    # ------------------------------------------------------------

    for path in train_files:
        try:
            data = load_json(path)
        except Exception as exc:
            errors.append(
                f"Malformed training JSON: {path.name}: {exc}"
            )
            continue

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
            errors.append(
                f"{path.name}: missing fields: {sorted(missing)}"
            )
            continue

        key = (
            data["algorithm"],
            data["environment"],
            data["seed"],
        )

        if key in train_runs:
            errors.append(
                f"Duplicate training result: {key}"
            )
        else:
            train_runs[key] = data

        if (data["algorithm"], data["environment"]) not in expected_combinations:
            errors.append(
                f"Invalid training combination: "
                f"{data['algorithm']} + {data['environment']}"
            )

        if data["seed"] not in expected_seeds:
            errors.append(
                f"Invalid training seed in {path.name}: "
                f"{data['seed']}"
            )
       
        expected_timesteps = environment_budgets.get(data["environment"])

        if expected_timesteps is None:
            errors.append(
                f"No training budget configured for environment "
                f"{data['environment']} in {path.name}"
            )
        elif data["timesteps"] != expected_timesteps:
            errors.append(
                f"Incorrect timesteps in {path.name}: "
                f"expected {expected_timesteps}, got {data['timesteps']}"
            )
     
        if data["device"] != config["training"]["device"]:
            errors.append(
                f"Incorrect device in {path.name}: "
                f"{data['device']}"
            )

        if data["training_time_seconds"] < 0:
            errors.append(
                f"Negative training time in {path.name}"
            )

    # ------------------------------------------------------------
    # Check evaluation results
    # ------------------------------------------------------------

    for path in eval_files:
        try:
            data = load_json(path)
        except Exception as exc:
            errors.append(
                f"Malformed evaluation JSON: {path.name}: {exc}"
            )
            continue

        required = {
            "algorithm",
            "environment",
            "episodes",
            "deterministic",
            "device",
            "training_seed",
            "evaluation_seeds",
            "episode_returns",
            "mean_return",
            "std_return",
        }

        missing = required - data.keys()

        if missing:
            errors.append(
                f"{path.name}: missing fields: {sorted(missing)}"
            )
            continue

        key = (
            data["algorithm"],
            data["environment"],
            data["training_seed"],
        )

        if key in eval_runs:
            errors.append(
                f"Duplicate evaluation result: {key}"
            )
        else:
            eval_runs[key] = data

        if (data["algorithm"], data["environment"]) not in expected_combinations:
            errors.append(
                f"Invalid evaluation combination: "
                f"{data['algorithm']} + {data['environment']}"
            )

        if data["training_seed"] not in expected_seeds:
            errors.append(
                f"Invalid evaluation seed in {path.name}: "
                f"{data['training_seed']}"
            )

        if data["episodes"] != episodes:
            errors.append(
                f"Incorrect episode count in {path.name}: "
                f"{data['episodes']}"
            )

        if data["deterministic"] is not config["evaluation"]["deterministic"]:
            errors.append(
                f"Evaluation deterministic setting is incorrect in "
                f"{path.name}"
            )

        if data["device"] != config["training"]["device"]:
            errors.append(
                f"Incorrect device in {path.name}: "
                f"{data['device']}"
            )

        if len(data["evaluation_seeds"]) != episodes:
            errors.append(
                f"Incorrect number of evaluation seeds in {path.name}"
            )

        if len(data["episode_returns"]) != episodes:
            errors.append(
                f"Incorrect number of episode returns in {path.name}"
            )

        if len(data["episode_returns"]) == episodes:
            calculated_mean = (
                sum(data["episode_returns"]) / episodes
            )

            if abs(calculated_mean - data["mean_return"]) > 1e-9:
                errors.append(
                    f"Incorrect mean return in {path.name}"
                )

    # ------------------------------------------------------------
    # Check correspondence between training and evaluation
    # ------------------------------------------------------------

    train_keys = set(train_runs)
    eval_keys = set(eval_runs)

    missing_train = eval_keys - train_keys
    missing_eval = train_keys - eval_keys

    for key in sorted(missing_train):
        errors.append(
            f"Evaluation exists without training result: {key}"
        )

    for key in sorted(missing_eval):
        errors.append(
            f"Training exists without evaluation result: {key}"
        )

    # ------------------------------------------------------------
    # Check expected number of runs
    # ------------------------------------------------------------

    if len(train_files) != expected_runs:
        errors.append(
            f"Expected {expected_runs} training files, "
            f"found {len(train_files)}"
        )

    if len(eval_files) != expected_runs:
        errors.append(
            f"Expected {expected_runs} evaluation files, "
            f"found {len(eval_files)}"
        )

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    print()
    print(f"RL Benchmark {config["benchmark"]["version"]} — Validation")
    print("=" * 40)
    print(f"Expected runs:         {expected_runs}")
    print(f"Training results:      {len(train_files)}")
    print(f"Evaluation results:    {len(eval_files)}")
    print(f"Expected combinations: {len(expected_combinations)}")
    print(f"Expected seeds:        {seeds}")
    print(f"Evaluation episodes:   {episodes}")
    print("Environment budgets:")
    for environment, budget in environment_budgets.items():
        print(f"  {environment}: {budget}")    
    print()

    if errors:
        print(f"Validation errors:     {len(errors)}")
        print()

        for error in errors:
            print(f"ERROR: {error}")

        print()
        print("Status: FAIL")
        raise SystemExit(1)

    print("Missing runs:          0")
    print("Duplicate runs:        0")
    print("Malformed files:       0")
    print("Invalid combinations:  0")
    print("Inconsistent results:  0")
    print()
    print("Status: PASS")


if __name__ == "__main__":
    main()
