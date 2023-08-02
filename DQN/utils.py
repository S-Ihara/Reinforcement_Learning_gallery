from typing import NamedTuple

class Configs(NamedTuple):
    """
    ハイパーパラメータ
    """
    env_name: str = "Breakout-v4"
    frame_stack: int = 3
    gray: bool = True
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 1024
    min_experiences: int = 2048
    memory_size: int = 10000
    num_episodes: int = 100
