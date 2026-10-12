# IIT Kanpur IDEAL Lab Server

## RL Benchmark v1.1

---

## Purpose

This benchmark provides a fixed reference setup for evaluating and comparing reinforcement learning algorithms on standard Gymnasium environments. Version 1.1 extends the v1.0 benchmark with additional Box2D-based environments while keeping the original v1.0 benchmark unchanged.

The reference results generated using this protocol are intended to remain frozen so that future RL algorithms and implementations can be compared against the same baseline.

---

## Benchmark Protocol

### Algorithms

The benchmark includes the following reinforcement learning algorithms:

* A2C
* DQN
* PPO
* DDPG
* TD3
* SAC

### Environments

#### Discrete

* `LunarLander-v3`

#### Continuous

* `LunarLanderContinuous-v3`
* `BipedalWalker-v3`

### Training

Training budgets are environment-specific:

| Environment                | Training Timesteps |
| -------------------------- | -----------------: |
| `LunarLander-v3`           |          1,000,000 |
| `LunarLanderContinuous-v3` |          1,000,000 |
| `BipedalWalker-v3`         |          5,000,000 |

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

Not every algorithm is applicable to every environment. The benchmark contains **13 valid algorithm-environment combinations**.

| Algorithm | Valid Environments                                               |
| --------- | ---------------------------------------------------------------- |
| A2C       | `LunarLander-v3`, `LunarLanderContinuous-v3`, `BipedalWalker-v3` |
| DQN       | `LunarLander-v3`                                                 |
| PPO       | `LunarLander-v3`, `LunarLanderContinuous-v3`, `BipedalWalker-v3` |
| DDPG      | `LunarLanderContinuous-v3`, `BipedalWalker-v3`                   |
| TD3       | `LunarLanderContinuous-v3`, `BipedalWalker-v3`                   |
| SAC       | `LunarLanderContinuous-v3`, `BipedalWalker-v3`                   |

With five random seeds for each valid combination, the benchmark consists of:

**13 algorithm-environment combinations × 5 seeds = 65 training runs**

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

There are **13 valid algorithm-environment combinations**. With five random seeds, the reference benchmark consists of:

* **65 training runs**
* **165,000,000 total training environment steps**
* **16,500 evaluation checkpoints**
* **825,000 evaluation episodes**

The number of evaluation checkpoints per run depends on the environment-specific training budget. Evaluation is performed every 10,000 training steps, with 50 evaluation episodes at each checkpoint.

### Running the Reference Benchmark

To run the entire benchmark, activate a python virtual environment and execute the following command from the v1.1 directory:

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
NUM_WORKERS = 30
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

* Training results for all 65 runs.
* Final deterministic evaluation results for all 65 runs.
* Complete evaluation curves for all 65 runs.
* Stable-Baselines3 training logs for all 65 runs.

Derived summaries are stored in:

```text
results/summaries/
```

The summary files include the following.

`training_dynamics.json`: Training-log dynamics for all 65 runs.

`evaluation_dynamics.json`: Evaluation dynamics aggregated across the five training seeds for each of the 13 algorithm-environment combinations.

`final_performance.json`: Final deterministic evaluation performance at the end of each environment-specific training budget.

`final_performance.csv`: Tabular version of the final performance results, including the individual seed-level final returns.

`training_time.json`: Wall-clock training-time statistics for the 65 reference runs.

`benchmark_summary.csv`: Combined final-performance and training-time summary for all 13 algorithm-environment combinations.

---

## Plots

Benchmark plots are stored in:

```text
plots/
```

The reference set contains **26 PNG plots**:

* 13 training-dynamics plots
* 13 deterministic-evaluation plots

The training plots show individual training seeds together with their cross-seed mean and standard deviation where data are available. The evaluation plots show the mean deterministic evaluation return across training seeds together with the cross-seed standard deviation.

---

## Directory Structure

```text
v1.1/
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
| `configs/`           | Benchmark configuration files defining the v1.1 experimental protocol and settings.                                                         |
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
| `README.md`          | Main documentation describing the v1.1 benchmark protocol, execution procedure, results, reproducibility, and directory structure.          |

---

## Reproducibility

The benchmark configuration for the v1.1 protocol is defined in:

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

If a new experiment is intended to be compared with the v1.1 reference benchmark, it should use the same protocol, including:

* The environment-specific training budget defined in `configs/benchmark.yaml`.
* CPU training.
* The specified algorithm-environment combination.
* One or more of the reference random seeds when exact replication is desired.
* Deterministic evaluation.
* 50 evaluation episodes.
* Evaluation checkpoints every 10,000 training steps.

Changes to the benchmark protocol should result in a new benchmark version rather than modification of v1.1. The v1.0 benchmark is maintained separately and must not be modified when conducting v1.1 experiments or new research experiments.

---

## Software

The benchmark uses:

| Software          | Version |
| ----------------- | ------- |
| Gymnasium         | 1.3.0   |
| Stable-Baselines3 | 2.9.0   |
| Box2D             | 2.3.10  |
| pygame-ce         | 2.5.8   |
| SWIG              | 4.2.0   |

The Box2D environments in v1.1 require the Box2D-related dependencies listed above.
