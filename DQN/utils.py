from typing import NamedTuple

class Configs(NamedTuple):
    """
    ハイパーパラメータ
    """
    env_name: str = "MiniGrid-Empty-Random-6x6-v0"
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 512
    min_experiences: int = 512
    memory_size: int = 10000
    num_episodes: int = 100
