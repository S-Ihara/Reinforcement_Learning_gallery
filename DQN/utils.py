from typing import NamedTuple, Optional

class Configs(NamedTuple):
    """
    ハイパーパラメータ
    """
    env_name: str = "Breakout-v4"
    frame_stack: int = 1
    gray: bool = True
    gamma: float = 0.96
    lr: float = 4e-4
    q_update_steps: int = 2
    target_update_steps: int = 500
    target_update_epochs: Optional[int] = None
    batch_size: int = 16
    min_experiences: int = 512
    memory_size: int = 100000
    num_episodes: int = 1000
