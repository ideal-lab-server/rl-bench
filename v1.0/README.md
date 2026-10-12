# IIT Kanpur IDEAL Lab Server

## RL Benchmark v1.0

---
## Purpose

This benchmark provides a fixed reference setup for evaluating and comparing reinforcement learning algorithms on standard Gymnasium environments. The reference results generated using this protocol are intended to remain **frozen** so that future RL algorithms and implementations can be compared against the same baseline.

---

## Benchmark Protocol

### Algorithms

* A2C
* DQN
* PPO
* DDPG
* TD3
* SAC

### Environments

#### Discrete

* `CartPole-v1`
* `Acrobot-v1`
* `MountainCar-v0`

#### Continuous

* `MountainCarContinuous-v0`
* `Pendulum-v1`

### Training

Training timesteps are environment-specific:

| Environment                | Training Timesteps |
| -------------------------- | -----------------: |
| `CartPole-v1`              |            100,000 |
| `Acrobot-v1`               |            500,000 |
| `MountainCar-v0`           |          1,000,000 |
| `MountainCarContinuous-v0` |            300,000 |
| `Pendulum-v1`              |            500,000 |

Additional training settings:

* **Number of random seeds:** 5
* **Training seeds:** `0, 1, 2, 3, 4`
* **Training device:** CPU

### Evaluation

* **Evaluation checkpoints:** Every 10,000 training steps
* **Evaluation episodes per checkpoint:** 50
* **Deterministic evaluation:** Yes
* **Primary metric:** Mean episode return

---

## Valid Algorithm-Environment Combinations

Not every algorithm is applicable to every environment. The benchmark contains **19 valid algorithm-environment combinations**.

| Algorithm | Valid Environments                                                                       |
| --------- | ---------------------------------------------------------------------------------------- |
| A2C       | `CartPole-v1`, `Acrobot-v1`, `MountainCar-v0`, `MountainCarContinuous-v0`, `Pendulum-v1` |
| DQN       | `CartPole-v1`, `Acrobot-v1`, `MountainCar-v0`                                            |
| PPO       | `CartPole-v1`, `Acrobot-v1`, `MountainCar-v0`, `MountainCarContinuous-v0`, `Pendulum-v1` |
| DDPG      | `MountainCarContinuous-v0`, `Pendulum-v1`                                                |
| TD3       | `MountainCarContinuous-v0`, `Pendulum-v1`                                                |
| SAC       | `MountainCarContinuous-v0`, `Pendulum-v1`                                                |

---

## Reported Metrics

For each training seed, the mean episode return is computed from the 50 deterministic evaluation episodes at each checkpoint. For each checkpoint, the five seed-level mean returns are then aggregated:

* **Mean return:** Mean of the five seed-level mean returns
* **Standard deviation:** Standard deviation of the five seed-level mean returns

The standard deviation therefore measures **variation across independent training seeds**, rather than variation across evaluation episodes. 

At each checkpoint, the evaluation dynamics are reported as:

> **Mean return ± standard deviation across seeds**


The **final benchmark result** is the value at the final evaluation checkpoint for each environment, corresponding to that environment's specified training budget.

The benchmark also records:

* Wall-clock training time for each run
* Individual final evaluation result for each training seed

Training rollout statistics are recorded separately in the Stable-Baselines3 training logs. These statistics are **not used as the primary benchmark metric**. In particular, **training rollout reward and deterministic evaluation reward should not be interpreted as the same measurement**.

---
## Reference Runs

There are 19 valid algorithm-environment combinations. With 5 random seeds, the reference benchmark consists of:

* **95 training runs**
* **44,000,000 total training environment steps**
* **4,400 evaluation checkpoints**
* **220,000 evaluation episodes**

The number of evaluation checkpoints per run depends on the environment-specific training budget. Evaluation is performed every 10,000 training steps, with 50 evaluation episodes at each checkpoint.

### Running the Reference Benchmark

Activate a python virtual environment and run the following command from the v1.0 benchmark directory:

```bash
PYTHONPATH=src python src/running/orchestrate.py
```

Running the above script:

* Removes previous results, models, plots, summaries, etc.
* Runs the entire benchmark from scratch.
* Saves the reference models in the appropriate folder.
* Analyses the obtained results.
* Generates summaries and plots.

> **Warning:** The orchestration script removes previous benchmark artifacts. Do not run it in the frozen reference directory unless the reference results are intentionally being regenerated.

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

* Training results for all 95 runs
* Final deterministic evaluation results for all 95 runs
* Complete evaluation curves for all 95 runs
* Stable-Baselines3 training logs for all 95 runs

### Derived Summaries

Derived summaries are stored in:

```text
results/summaries/
```

The summary files are:

`training_dynamics.json`: 
Training-log dynamics for all 95 runs.

`evaluation_dynamics.json`: 
Evaluation dynamics aggregated across the five training seeds for each of the 19 algorithm-environment combinations.

`final_performance.json`:
Final deterministic evaluation performance at the end of each environment-specific training budget.

`final_performance.csv`:
Tabular version of the final performance results, including the individual seed-level final returns.

`training_time.json`:
Wall-clock training-time statistics for the 95 reference runs.

`benchmark_summary.csv`:
Combined final-performance and training-time summary for all 19 algorithm-environment combinations.

---

## Plots

Plots are stored in:

```text
plots/
```

The current reference set contains **38 PNG plots**:

* **19 training-dynamics plots**: The training plots show individual training seeds together with their cross-seed mean and standard deviation where data are available.
* **19 deterministic-evaluation plots**: The evaluation plots show the mean deterministic evaluation return across training seeds together with the cross-seed standard deviation.

---

## Directory Structure

```text
v1.0/
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

| Directory / File     | Description                                                                                                                                                   |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `configs/`           | Benchmark configuration files defining the experimental protocol and settings.                                                                                |
| `docs/`              | Additional documentation related to the benchmark.                                                                                                            |
| `logs/`              | Training logs generated for individual algorithm–environment–seed runs. Each training run has its own subdirectory containing the corresponding training log. |
| `models/reference/`  | Reference trained models corresponding to the benchmark runs.                                                                                                 |
| `plots/`             | Figures and plots generated from training and evaluation results.                                                                                             |
| `results/reference/` | Detailed per-run results, including training, evaluation, and learning-curve data.                                                                            |
| `results/summaries/` | Aggregated benchmark summaries and final performance statistics.                                                                                              |
| `src/analysis/`      | Scripts for analyzing training logs, evaluation results, final performance, and training time.                                                                |
| `src/conversion/`    | Utilities for converting benchmark results between data formats.                                                                                              |
| `src/plotting/`      | Scripts for generating benchmark plots and figures.                                                                                                           |
| `src/running/`       | Scripts for training models and orchestrating benchmark experiments.                                                                                          |
| `src/utilities/`     | Common utilities used by the benchmark scripts.                                                                                                               |
| `src/validation/`    | Scripts for validating benchmark results and checking result consistency.                                                                                     |
| `README.md`          | Main documentation describing the benchmark protocol, execution procedure, results, and directory structure.                                                  |


---

## Reproducibility

The benchmark configuration for v1.0 protocol is defined in:

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

If a new experiment is intended to be compared with the v1.0 reference benchmark, it should use the same protocol, including:

* The environment-specific training budget defined in `configs/benchmark.yaml`.
* CPU training.
* The specified algorithm-environment combination.
* One or more of the reference random seeds when exact replication is desired.
* Deterministic evaluation.
* 50 evaluation episodes.
* Evaluation checkpoints every 10,000 training steps.

Changes to the benchmark protocol should result in a **new benchmark version** rather than modification of v1.0.

---

## Software

The benchmark uses:

| Software          | Version |
| ----------------- | ------- |
| Gymnasium         | 1.3.0   |
| Stable-Baselines3 | 2.9.0   |
