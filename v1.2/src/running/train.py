import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from utilities.benchmark_utils import is_valid_combination
from utilities.results_utils import save_json


ALGORITHMS = {
    "A2C": A2C,
    "PPO": PPO,
    "DDPG": DDPG,
    "TD3": TD3,
    "SAC": SAC,
}


class EvaluationCallback(BaseCallback):
    """Evaluate the policy at fixed training-step intervals."""

    def __init__(
        self,
        algorithm,
        environment,
        frequency,
        episodes,
        training_seed,
        curve_path,
    ):
        super().__init__(verbose=0)

        self.algorithm = algorithm
        self.environment = environment
        self.frequency = frequency
        self.episodes = episodes
        self.training_seed = training_seed
        self.curve_path = Path(curve_path)

        self.evaluation_env = None
        self.evaluations = []
        self.next_evaluation = frequency

    def _init_callback(self):
        self.evaluation_env = gym.make(self.environment)

    def _on_step(self):
        if self.num_timesteps >= self.next_evaluation:
            self._evaluate()
            self.next_evaluation += self.frequency

        return True

    def _evaluate(self):
        returns = []
        evaluation_seeds = []

        checkpoint = self.num_timesteps // self.frequency

        for episode in range(self.episodes):
            evaluation_seed = (
                100000
                + self.training_seed * 10000
                + checkpoint * 100
                + episode
            )

            evaluation_seeds.append(evaluation_seed)

            observation, info = self.evaluation_env.reset(
                seed=evaluation_seed
            )

            terminated = False
            truncated = False
            episode_return = 0.0

            while not (terminated or truncated):
                action, _ = self.model.predict(
                    observation,
                    deterministic=True,
                )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = self.evaluation_env.step(action)

                episode_return += reward

            returns.append(episode_return)

        returns = np.asarray(returns)
        episode_returns = returns.tolist()

        result = {
            "algorithm": self.algorithm,
            "environment": self.environment,
            "training_seed": self.training_seed,
            "timesteps": self.num_timesteps,
            "episodes": self.episodes,
            "deterministic": True,
            "evaluation_seeds": evaluation_seeds,
            "episode_returns": episode_returns,
            "mean_return": sum(episode_returns) / len(episode_returns),
            "std_return": float(returns.std()),
        }

        self.evaluations.append(result)

        save_json(
            self.evaluations,
            self.curve_path,
        )

        print(
            f"Evaluation at {self.num_timesteps} steps: "
            f"mean={returns.mean():.2f}, "
            f"std={returns.std():.2f}"
        )

    def _on_training_end(self):
        if self.evaluation_env is not None:
            self.evaluation_env.close()


def train(
    algorithm,
    environment,
    seed,
    timesteps,
    evaluation_frequency,
    evaluation_episodes,
    log_directory,
    curve_path,
    output_path,
    result_path,
):
    if not is_valid_combination(algorithm, environment):
        raise ValueError(
            f"Invalid algorithm/environment combination: "
            f"{algorithm} + {environment}"
        )

    log_directory = Path(log_directory)
    log_directory.mkdir(parents=True, exist_ok=True)

    # Prevent SB3 from using /tmp/SB3-* if any internal
    # logger configuration falls back to environment variables.
    os.environ["SB3_LOGDIR"] = str(log_directory)
    os.environ["SB3_LOG_FORMAT"] = "stdout,csv"

    env = gym.make(environment)
    env.reset(seed=seed)

    model_class = ALGORITHMS[algorithm]

    model = model_class(
        "MlpPolicy",
        env,
        seed=seed,
        device="cpu",
        verbose=0,
    )

    # Explicit dedicated logger.
    logger = configure(
        str(log_directory),
        ["csv"],
    )
    model.set_logger(logger)

    callback = EvaluationCallback(
        algorithm=algorithm,
        environment=environment,
        frequency=evaluation_frequency,
        episodes=evaluation_episodes,
        training_seed=seed,
        curve_path=curve_path,
    )

    start_time = time.perf_counter()

    model.learn(
        total_timesteps=timesteps,
        callback=callback,
    )

    training_time = time.perf_counter() - start_time

    model.save(output_path)

    env.close()

    result = {
        "algorithm": algorithm,
        "environment": environment,
        "seed": seed,
        "timesteps": timesteps,
        "device": "cpu",
        "training_time_seconds": training_time,
        "model_path": str(output_path),
        "learning_curve_path": str(curve_path),
        "log_directory": str(log_directory),
    }

    save_json(result, result_path)

    print(f"Algorithm:       {algorithm}")
    print(f"Environment:     {environment}")
    print(f"Seed:             {seed}")
    print(f"Timesteps:        {timesteps}")
    print(f"Training time:    {training_time:.2f} seconds")
    print(f"Model saved to:   {output_path}")
    print(f"Learning curve:   {curve_path}")
    print(f"SB3 logs:         {log_directory}")
    print(f"Result saved to:  {result_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train one RL benchmark run."
    )

    parser.add_argument("--algorithm", required=True, choices=ALGORITHMS)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--timesteps", required=True, type=int)
    parser.add_argument(
        "--evaluation-frequency",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--evaluation-episodes",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--log-directory",
        required=True,
    )
    parser.add_argument(
        "--curve",
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--result", required=True)

    args = parser.parse_args()

    train(
        algorithm=args.algorithm,
        environment=args.environment,
        seed=args.seed,
        timesteps=args.timesteps,
        evaluation_frequency=args.evaluation_frequency,
        evaluation_episodes=args.evaluation_episodes,
        log_directory=args.log_directory,
        curve_path=args.curve,
        output_path=args.output,
        result_path=args.result,
    )


if __name__ == "__main__":
    main()
