ALGORITHM_ENVIRONMENTS = {
    "A2C": [
        "CartPole-v1",
        "Acrobot-v1",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
        "Pendulum-v1",
    ],
    "DQN": [
        "CartPole-v1",
        "Acrobot-v1",
        "MountainCar-v0",
    ],
    "PPO": [
        "CartPole-v1",
        "Acrobot-v1",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
        "Pendulum-v1",
    ],
    "DDPG": [
        "MountainCarContinuous-v0",
        "Pendulum-v1",
    ],
    "TD3": [
        "MountainCarContinuous-v0",
        "Pendulum-v1",
    ],
    "SAC": [
        "MountainCarContinuous-v0",
        "Pendulum-v1",
    ],
}


def get_valid_environments(algorithm):
    """Return environments supported by the specified algorithm."""
    if algorithm not in ALGORITHM_ENVIRONMENTS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return ALGORITHM_ENVIRONMENTS[algorithm]


def is_valid_combination(algorithm, environment):
    """Return True if the algorithm/environment combination is valid."""
    return environment in get_valid_environments(algorithm)


def get_valid_combinations():
    """Return all valid algorithm/environment combinations."""
    return [
        (algorithm, environment)
        for algorithm, environments in ALGORITHM_ENVIRONMENTS.items()
        for environment in environments
    ]
