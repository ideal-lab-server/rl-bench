# IIT Kanpur IDEAL Lab Server

## RL Benchmark v1.2

---

## Purpose

This benchmark provides a fixed reference setup for evaluating and comparing reinforcement learning algorithms on standard Gymnasium environments. Version 1.2 extends the benchmark to MuJoCo-based continuous-control environments while keeping the v1.0 and v1.1 benchmarks unchanged.

The reference results generated using this protocol are intended to remain frozen so that future RL algorithms and implementations can be compared against the same baseline.

---

## Benchmark Protocol

### Algorithms

The benchmark includes the following reinforcement learning algorithms:

* A2C
* PPO
* DDPG
* TD3
* SAC

### Environments

All v1.2 environments have continuous action spaces.

* `Swimmer-v5`

* `Hopper-v5`

* `Walker2d-v5`

* `HalfCheetah-v5`

* `Ant-v5`

* `Humanoid-v5`

### Training

Training budgets are environment-specific:

| Environment      | Training Timesteps |
| ---------------- | -----------------: |
| `Swimmer-v5`     |            500,000 |
| `Hopper-v5`      |          1,000,000 |
| `Walker2d-v5`    |          1,000,000 |
| `HalfCheetah-v5` |          1,000,000 |
| `Ant-v5`         |          2,000,000 |
| `Humanoid-v5`    |          5,000,000 |

Additional training settings:

* Number of random seeds: **5**
* Training seeds: **0, 1, 2, 3, 4**
* Training device: **CPU**

### Evaluation

* Evaluation checkpoints: Every **10,000 training steps**
* Evaluation episodes per checkpoint: **50**
* Deterministic evaluation: **Yes**
* Primary metric: **Mean episode return**

---

## Valid Algorithm-Environment Combinations

All five algorithms are applicable to all six v1.2 MuJoCo environments. The benchmark therefore contains **30 valid algorithm-environment combinations**.

| Algorithm | Valid Environments |
| --------- | ------------------ |
| A2C       | `Swimmer-v5`, `Hopper-v5`, `Walker2d-v5`, `HalfCheetah-v5`, `Ant-v5`, `Humanoid-v5` |
| PPO       | `Swimmer-v5`, `Hopper-v5`, `Walker2d-v5`, `HalfCheetah-v5`, `Ant-v5`, `Humanoid-v5` |
| DDPG      | `Swimmer-v5`, `Hopper-v5`, `Walker2d-v5`, `HalfCheetah-v5`, `Ant-v5`, `Humanoid-v5` |
| TD3       | `Swimmer-v5`, `Hopper-v5`, `Walker2d-v5`, `HalfCheetah-v5`, `Ant-v5`, `Humanoid-v5` |
| SAC       | `Swimmer-v5`, `Hopper-v5`, `Walker2d-v5`, `HalfCheetah-v5`, `Ant-v5`, `Humanoid-v5` |

With five random seeds for each valid combination, the benchmark consists of:

**30 algorithm-environment combinations × 5 seeds = 150 training runs**

---

## Reported Metrics

For each training seed, the mean episode return is computed from the 50 deterministic evaluation episodes at each checkpoint. For each checkpoint, the five seed-level mean returns are then aggregated:

$$
\text{Mean return}=\text{mean of the five seed-level mean returns}
$$
$$
\text{Standard deviation}=\text{standard deviation of the five seed-level mean returns}
$$

The standard deviation therefore measures variation across independent training seeds, rather than variation across evaluation episodes. The evaluation dynamics are reported as:

$$
\text{mean return} \pm \text{standard deviation}
$$

across the five training seeds at each evaluation checkpoint. The final benchmark result is the value at the final evaluation checkpoint for each environment, corresponding to that environment's specified training budget.

The benchmark also records:

* Wall-clock training time for each run.
* Individual final evaluation results for each training seed.

Training rollout statistics are recorded separately in the Stable-Baselines3 training logs. These statistics are not used as the primary benchmark metric. In particular, training rollout reward and deterministic evaluation reward should not be interpreted as the same measurement.

---

## Reference Runs

There are **30 valid algorithm-environment combinations**. With five random seeds, the reference benchmark consists of:

* **150 training runs**

* **262,500,000 total training environment steps**

* **26,250 evaluation checkpoints**

* **1,312,500 evaluation episodes**

The number of evaluation checkpoints per run depends on the environment-specific training budget. Evaluation is performed every 10,000 training steps, with 50 evaluation episodes at each checkpoint.

### Running the Reference Benchmark

To run the entire benchmark, activate a python virtual environment and execute the following command from the v1.2 directory:

```bash
PYTHONPATH=src python src/running/orchestrate.py
```

Running the above script:

* Removes previous results, models, plots, summaries, and related generated artifacts.
* Runs the entire benchmark from scratch.
* Saves the reference models in the appropriate directory.
* Analyzes the obtained results.
* Generates summaries and plots.

> **Warning:** The orchestration script is intended for generating the complete benchmark from scratch. Do not run it in the reference directory if the existing reference artifacts must be preserved.

## Parallel Computation

The benchmark executes independent training runs in parallel to reduce the total wall-clock time. The parallel execution is implemented in `src/running/run_benchmark.py` using a worker pool with the following configuration:

```python
NUM_WORKERS = 10
THREADS_PER_WORKER = 1
```

 Each worker executes one complete algorithm-environment-seed training run at a time. The benchmark does not parallelize the training steps within a single run. Instead, **different runs are executed concurrently**. This preserves the independence of the training seeds while making use of the available CPU resources.

The parallel workers are applied only to the training stage. Benchmark validation, result analysis, summary generation, CSV conversion, and plot generation are performed after all training runs have completed.

---

## Results and Analysis

The frozen reference results are stored in:

```text
results/reference/
```

These include:

* Training results for all 150 runs.
* Final deterministic evaluation results for all 150 runs.
* Complete evaluation curves for all 150 runs.
* Stable-Baselines3 training logs for all 150 runs.

Derived summaries are stored in:

```text
results/summaries/
```

The summary files include the following.

`training_dynamics.json`: Training-log dynamics for all 150 runs.

`evaluation_dynamics.json`: Evaluation dynamics aggregated across the five training seeds for each of the 30 algorithm-environment combinations.

`final_performance.json`: Final deterministic evaluation performance at the end of each environment-specific training budget.

`final_performance.csv`: Tabular version of the final performance results, including the individual seed-level final returns.

`training_time.json`: Wall-clock training-time statistics for the 150 reference runs.

`benchmark_summary.csv`: Combined final-performance and training-time summary for all 30 algorithm-environment combinations.

---

## Plots

Benchmark plots are stored in:

```text
plots/
```

The reference set contains **60 PNG plots**:

* 30 training-dynamics plots
* 30 deterministic-evaluation plots

The training plots show individual training seeds together with their cross-seed mean and standard deviation where data are available. The evaluation plots show the mean deterministic evaluation return across training seeds together with the cross-seed standard deviation.

---

## Directory Structure

```text
v1.2/
├── configs/
├── docs/
├── logs/
├── models/
│   └── reference/
├── plots/
├── results/
│   ├── reference/
│   └── summaries/
├── src/
│   ├── analysis/
│   ├── conversion/
│   ├── plotting/
│   ├── running/
│   ├── utilities/
│   └── validation/
└── README.md
```

### Directory Description

| Directory / File     | Description                                                                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `configs/`           | Benchmark configuration files defining the v1.2 experimental protocol and settings.                                                         |
| `docs/`              | Additional documentation related to the benchmark.                                                                                          |
| `logs/`              | Training logs generated for individual algorithm-environment-seed runs.                                                                     |
| `models/reference/`  | Reference trained models corresponding to the benchmark runs.                                                                               |
| `plots/`             | Figures and plots generated from training and evaluation results.                                                                           |
| `results/reference/` | Detailed per-run reference results, including training results, deterministic evaluation results, evaluation curves, and training-log data. |
| `results/summaries/` | Aggregated benchmark summaries, final performance statistics, and training-time statistics.                                                 |
| `src/analysis/`      | Scripts for analyzing training logs, evaluation results, final performance, and training time.                                              |
| `src/conversion/`    | Utilities for converting benchmark results between data formats.                                                                            |
| `src/plotting/`      | Scripts for generating benchmark plots and figures.                                                                                         |
| `src/running/`       | Scripts for training models and orchestrating benchmark experiments.                                                                        |
| `src/utilities/`     | Common utilities used by the benchmark scripts.                                                                                             |
| `src/validation/`    | Scripts for validating benchmark results and checking result consistency.                                                                   |
| `README.md`          | Main documentation describing the v1.2 benchmark protocol, execution procedure, results, reproducibility, and directory structure.          |

---

## Reproducibility

The benchmark configuration for the v1.2 protocol is defined in:

```text
configs/benchmark.yaml
```

The reference results in:

```text
results/reference/
```

and the derived summaries in:

```text
results/summaries/
```

should be treated as **frozen reference artifacts**.

### Important

Do **not** modify or overwrite the frozen reference results when conducting new experiments. New experiments should be performed **outside this shared benchmark directory**, preferably in the researcher's personal project directory.

If a new experiment is intended to be compared with the v1.2 reference benchmark, it should use the same protocol, including:

* The environment-specific training budget defined in `configs/benchmark.yaml`.
* CPU training.
* The specified algorithm-environment combination.
* One or more of the reference random seeds when exact replication is desired.
* Deterministic evaluation.
* 50 evaluation episodes.
* Evaluation checkpoints every 10,000 training steps.

Changes to the benchmark protocol should result in a new benchmark version rather than modification of v1.2. The v1.0 and v1.1 benchmarks are maintained separately and must not be modified when conducting v1.2 experiments or new research experiments.

---

## Software

The benchmark uses:

| Software          | Version |
| ----------------- | ------- |
| Gymnasium         | 1.3.0   |
| Stable-Baselines3 | 2.9.0   |
| MuJoCo            | 3.14.0  |

## MuJoCo Environments

The v1.2 benchmark uses the MuJoCo environments provided by Gymnasium. All six environments use continuous action spaces and are therefore evaluated with A2C, PPO, DDPG, TD3, and SAC. DQN is not included in v1.2 because it is designed for discrete action spaces.
