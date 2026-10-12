ALGORITHM_ENVIRONMENTS = {
    "A2C": [
        "Swimmer-v5",
        "Hopper-v5",
        "Walker2d-v5",
        "HalfCheetah-v5",
        "Ant-v5",
        "Humanoid-v5",
    ],
    "PPO": [
        "Swimmer-v5",
        "Hopper-v5",
        "Walker2d-v5",
        "HalfCheetah-v5",
        "Ant-v5",
        "Humanoid-v5",
    ],
    "DDPG": [
        "Swimmer-v5",
        "Hopper-v5",
        "Walker2d-v5",
        "HalfCheetah-v5",
        "Ant-v5",
        "Humanoid-v5",
    ],
    "TD3": [
        "Swimmer-v5",
        "Hopper-v5",
        "Walker2d-v5",
        "HalfCheetah-v5",
        "Ant-v5",
        "Humanoid-v5",
    ],
    "SAC": [
        "Swimmer-v5",
        "Hopper-v5",
        "Walker2d-v5",
        "HalfCheetah-v5",
        "Ant-v5",
        "Humanoid-v5",
    ],
}


def is_valid_combination(algorithm, environment):
    """Return True if the algorithm-environment combination is valid."""
    return environment in ALGORITHM_ENVIRONMENTS.get(algorithm, [])


def get_valid_combinations():
    """Return all valid algorithm-environment combinations."""
    return [
        (algorithm, environment)
        for algorithm, environments in ALGORITHM_ENVIRONMENTS.items()
        for environment in environments
    ]
