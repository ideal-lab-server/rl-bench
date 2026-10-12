import os
import subprocess
import sys
from pathlib import Path
import yaml

def run_command(description, command, benchmark_root):
    """Run a command and stop the pipeline if it fails."""
    print()
    print("=" * 72)
    print(description)
    print("=" * 72)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(benchmark_root / "src")

    result = subprocess.run(
        command,
        cwd=benchmark_root,
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Step failed: {description} "
            f"(exit code {result.returncode})"
        )


def clean_generated_outputs(benchmark_root):
    """Delete contents of generated-output directories while preserving them."""
    directories = [
        benchmark_root / "results" / "reference",
        benchmark_root / "results" / "summaries",
        benchmark_root / "models" / "reference",
        benchmark_root / "logs",
        benchmark_root / "plots",
    ]

    print()
    print("=" * 72)
    print("Cleaning previous benchmark outputs")
    print("=" * 72)

    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Expected directory does not exist: {directory}"
            )

        print(f"Cleaning: {directory}")

        subprocess.run(
            [
                "find",
                str(directory),
                "-mindepth",
                "1",
                "-delete",
            ],
            check=True,
        )

    print("Cleanup completed.")


def main():
    benchmark_root = Path(__file__).resolve().parent.parent.parent
    src_dir = benchmark_root / "src"

    config_path = benchmark_root / "configs" / "benchmark.yaml"
    
    with config_path.open() as file:
        config = yaml.safe_load(file)

    benchmark_version = config["benchmark"]["version"]

    python = sys.executable

    # ------------------------------------------------------------------
    # 1. Clean previous generated outputs
    # ------------------------------------------------------------------
    clean_generated_outputs(benchmark_root)

    # ------------------------------------------------------------------
    # 2. Run the complete benchmark
    # ------------------------------------------------------------------
    run_command(
        f"Running RL Benchmark {benchmark_version}",
        [
            python,
            str(src_dir / "running" / "run_benchmark.py"),
            "--config",
            str(config_path),
        ],
        benchmark_root,
    )

    # ------------------------------------------------------------------
    # 3. Validate benchmark results before analysis
    # ------------------------------------------------------------------
    run_command(
        "Validating benchmark results",
        [
            python,
            str(src_dir / "validation" / "validate_results.py"),
        ],
        benchmark_root,
    )

    # ------------------------------------------------------------------
    # 4. Generate JSON analysis summaries
    # ------------------------------------------------------------------
    run_command(
        "Analyzing training dynamics",
        [
            python,
            str(src_dir / "analysis" / "analyze_training_logs.py"),
        ],
        benchmark_root,
    )

    run_command(
        "Analyzing evaluation dynamics",
        [
            python,
            str(src_dir / "analysis" / "analyze_evaluation.py"),
        ],
        benchmark_root,
    )

    run_command(
        "Analyzing final performance",
        [
            python,
            str(src_dir / "analysis" / "analyze_final_performance.py"),
        ],
        benchmark_root,
    )

    run_command(
        "Analyzing training time",
        [
            python,
            str(src_dir / "analysis" / "analyze_training_time.py"),
        ],
        benchmark_root,
    )

    # ------------------------------------------------------------------
    # 5. Convert JSON summaries to CSV
    # ------------------------------------------------------------------
    summaries_dir = benchmark_root / "results" / "summaries"

    csv_files = {
        "final_performance.json": "final_performance.csv",
        "training_time.json": "training_time.csv",
        "evaluation_dynamics.json": "evaluation_dynamics.csv",
        "training_dynamics.json": "training_dynamics.csv",
    }

    for json_name, csv_name in csv_files.items():
        run_command(
            f"Converting {json_name} to CSV",
            [
                python,
                str(src_dir / "conversion" / "json_to_csv.py"),
                "--input",
                str(summaries_dir / json_name),
                "--output",
                str(summaries_dir / csv_name),
            ],
            benchmark_root,
        )

    # ------------------------------------------------------------------
    # 6. Generate benchmark plots
    # ------------------------------------------------------------------
    run_command(
        "Generating benchmark plots",
        [
            python,
            str(src_dir / "plotting" / "plot_benchmark.py"),
        ],
        benchmark_root,
    )

    # ------------------------------------------------------------------
    # 7. Final validation
    # ------------------------------------------------------------------
    run_command(
        "Final validation",
        [
            python,
            str(src_dir / "validation" / "validate_results.py"),
        ],
        benchmark_root,
    )

    print()
    print("=" * 72)
    print(f"RL Benchmark {benchmark_version} completed successfully.")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("=" * 72)
        print("RL Benchmark v1.2 FAILED")
        print("=" * 72)
        print(error)
        sys.exit(1)
