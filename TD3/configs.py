from typing import NamedTuple, Optional

class Pendulum(NamedTuple):
    """
    ハイパーパラメータ
    """
    env_name: str = "BipedalWalker-v3"
    frame_stack: int = 1
    hidden_dim: int = 64
    activation: str = "ReLU"
    lr: float = 1e-4
    gamma: float = 0.98
    update_steps: int = 2
    tau: float = 0.001
    noise: float = 0.2
    batch_size: int = 64
    min_experiences: int = 2000
    memory_size: int = 50000
    num_episodes: int = 300
    seed: int = 42

class BipedalWalker(NamedTuple):
    env_name: str = "BipedalWalker-v3"
    frame_stack: int = 1
    hidden_dim: int = 64
    activation: str = "ReLU"
    lr: float = 1e-4
    gamma: float = 0.98
    update_steps: int = 1
    tau: float = 0.001
    noise: float = 0.1
    batch_size: int = 128
    min_experiences: int = 2000
    memory_size: int = 200000
    num_episodes: int = 1500
    seed: int = 42

class DefaultConfigs(NamedTuple):
    """
    ハイパーパラメータ
    """
    env_name: str = "BipedalWalkerHardcore-v3" # "Pendulum-v1" # "BipedalWalker-v3"
    frame_stack: int = 1
    #gray: bool = True
    #critic_hidden_dim: int = 64
    #actor_hidden_dim: int = 64
    hidden_dim: int = 128
    activation: str = "ReLU"
    lr: float = 1e-4
    gamma: float = 0.98
    update_steps: int = 1
    tau: float = 0.001
    noise: float = 0.1
    batch_size: int = 128
    min_experiences: int = 2000
    memory_size: int = 300000
    num_episodes: int = 5000
    seed: int = 42