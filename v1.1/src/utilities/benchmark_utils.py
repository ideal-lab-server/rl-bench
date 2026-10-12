ALGORITHM_ENVIRONMENTS = {

    "A2C": [
        "LunarLander-v3",
        "LunarLanderContinuous-v3",
        "BipedalWalker-v3",
    ],

    "DQN": [
        "LunarLander-v3",
    ],

    "PPO": [
        "LunarLander-v3",
        "LunarLanderContinuous-v3",
        "BipedalWalker-v3",
    ],

    "DDPG": [
        "LunarLanderContinuous-v3",
        "BipedalWalker-v3",
    ],

    "TD3": [
        "LunarLanderContinuous-v3",
        "BipedalWalker-v3",
    ],

    "SAC": [
        "LunarLanderContinuous-v3",
        "BipedalWalker-v3",
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
