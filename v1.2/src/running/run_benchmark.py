import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from utilities.benchmark_utils import get_valid_combinations


# ------------------------------------------------------------
# Parallel execution settings
# ------------------------------------------------------------

NUM_WORKERS = 10
THREADS_PER_WORKER = 1


def load_config(config_path):
    with open(config_path) as file:
        return yaml.safe_load(file)


def run_command(command):
    print("\nRunning:")
    print(" ".join(str(x) for x in command))
    print()

    env = os.environ.copy()

    env["OMP_NUM_THREADS"] = str(THREADS_PER_WORKER)
    env["MKL_NUM_THREADS"] = str(THREADS_PER_WORKER)
    env["OPENBLAS_NUM_THREADS"] = str(THREADS_PER_WORKER)
    env["NUMEXPR_NUM_THREADS"] = str(THREADS_PER_WORKER)

    subprocess.run(command, check=True, env=env)


def run_single(
    algorithm,
    environment,
    seed,
    timesteps,
    evaluation_frequency,
    evaluation_episodes,
    src_dir,
    results_dir,
    models_dir,
    logs_dir,
):
    run_name = f"{algorithm}_{environment}_seed{seed}"

    model_path = models_dir / run_name
    training_result = results_dir / f"{run_name}_train.json"
    evaluation_result = results_dir / f"{run_name}_eval.json"

    log_directory = logs_dir / run_name
    curve_path = results_dir / f"{run_name}_curve.json"

    log_directory.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(
        f"Starting: {algorithm} | {environment} | seed {seed}"
    )
    print("=" * 70)

    train_command = [
        "python",
        str(src_dir / "running" / "train.py"),
        "--algorithm",
        algorithm,
        "--environment",
        environment,
        "--seed",
        str(seed),
        "--timesteps",
        str(timesteps),
        "--evaluation-frequency",
        str(evaluation_frequency),
        "--evaluation-episodes",
        str(evaluation_episodes),
        "--log-directory",
        str(log_directory),
        "--curve",
        str(curve_path),
        "--output",
        str(model_path),
        "--result",
        str(training_result),
    ]

    run_command(train_command)

    # The final evaluation at the end of training is already performed
    # by the evaluation callback. Create the conventional
    # final evaluation JSON from that checkpoint.
    evaluation_command = [
        "python",
        str(src_dir / "analysis" / "extract_final_evaluation.py"),
        "--curve",
        str(curve_path),
        "--result",
        str(evaluation_result),
    ]

    run_command(evaluation_command)

    print(
        f"Completed: {algorithm} | {environment} | seed {seed}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the RL reference benchmark."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to benchmark configuration.",
    )

    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    src_dir = Path(__file__).resolve().parent.parent
    benchmark_root = src_dir.parent

    results_dir = benchmark_root / "results" / "reference"
    models_dir = benchmark_root / "models" / "reference"

    logging_config = config["logging"]
    logs_dir = benchmark_root / logging_config["directory"]

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    environment_budgets = config["environment_budget"]
    seeds = config["training"]["seeds"]

    evaluation_frequency = logging_config["frequency"]
    evaluation_episodes = config["evaluation"]["episodes"]

    requested_algorithms = config["algorithms"]

    requested_environments = (
        config["environments"]["discrete"]
        + config["environments"]["continuous"]
    )

    valid_combinations = set(get_valid_combinations())

    combinations = [
        (algorithm, environment)
        for algorithm in requested_algorithms
        for environment in requested_environments
        if (algorithm, environment) in valid_combinations
    ]

    runs = [
        (algorithm, environment, seed)
        for algorithm, environment in combinations
        for seed in range(seeds)
    ]

    print("RL Benchmark")
    print("============")
    print(f"Requested algorithms: {len(requested_algorithms)}")
    print(f"Requested environments: {len(requested_environments)}")
    print(f"Valid combinations: {len(combinations)}")
    print(f"Seeds:                {seeds}")
    print("Environment budgets:")
    for environment, budget in environment_budgets.items():
        print(f"  {environment}: {budget}")
    print(f"Evaluation frequency: {evaluation_frequency}")
    print(f"Evaluation episodes:  {evaluation_episodes}")
    print(f"Parallel workers:     {NUM_WORKERS}")
    print(f"Threads/worker:       {THREADS_PER_WORKER}")
    print(f"Total runs:           {len(runs)}")
    print(f"Models:               {models_dir}")
    print(f"Logs:                 {logs_dir}")
    print()

    with ThreadPoolExecutor(
        max_workers=NUM_WORKERS
    ) as executor:

        futures = [
            executor.submit(
                run_single,
                algorithm,
                environment,
                seed,
                environment_budgets[environment],
                evaluation_frequency,
                evaluation_episodes,
                src_dir,
                results_dir,
                models_dir,
                logs_dir,
            )
            for algorithm, environment, seed in runs
        ]

        for future in as_completed(futures):
            future.result()

    print()
    print("Benchmark completed successfully.")


if __name__ == "__main__":
    main()
