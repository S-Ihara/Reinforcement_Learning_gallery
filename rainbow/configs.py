from typing import NamedTuple, Optional

class DefaultConfigs(NamedTuple):
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
    batch_size: int = 128
    min_experiences: int = 512
    memory_size: int = 100000
    num_episodes: int = 3000
    seed: int = 42
    # 以下rainbow用のフラグ
    double: bool = True

class CartPole(NamedTuple):
    """CartPole-v1
    """
    env_name: str = "CartPole-v1"
    frame_stack: int = 1
    gamma: float = 0.96
    lr: float = 3e-4
    q_update_steps: int = 1
    target_update_steps: int = 500
    target_update_epochs: Optional[int] = None
    batch_size: int = 256
    min_experiences: int = 512
    memory_size: int = 100000
    num_episodes: int = 3000
    seed: int = 42

class Acrobot(NamedTuple):
    env_name: str = "Acrobot-v1"
    frame_stack: int = 1
    gamma: float = 0.95
    lr: float = 4e-4
    q_update_steps: int = 1
    target_update_steps: int = 3000
    target_update_epochs: Optional[int] = None
    batch_size: int = 64
    min_experiences: int = 1024
    memory_size: int = 100000
    num_episodes: int = 3000
    seed: int = 42
    reward_clip: bool = True

class Atari(NamedTuple):
    env_name: str = "Breakout-v4"
    frame_stack: int = 3
    gray: bool = True
    gamma: float = 0.99
    lr: float = 3e-4
    q_update_steps: int = 1
    target_update_steps: int = 1000
    target_update_epochs: Optional[int] = None
    batch_size: int = 512
    min_experiences: int = 2048
    memory_size: int = 300000
    num_episodes: int = 1000
    seed: int = 42
    reward_clip: bool = True
    double: bool = True

class Breakout(NamedTuple):
    env_name: str = "Breakout-v4"
    frame_stack: int = 3
    gray: bool = True
    gamma: float = 0.99
    lr: float = 3e-4
    q_update_steps: int = 1
    target_update_steps: int = 1000
    target_update_epochs: Optional[int] = None
    batch_size: int = 512
    min_experiences: int = 2048
    memory_size: int = 300000
    num_episodes: int = 3000
    seed: int = 42

class MiniGridEmpty(NamedTuple):
    env_name: str = "MiniGrid-Empty-Random-6x6-v0"
    tile_size: int = 8
    frame_stack: int = 1
    gamma: float = 0.99
    lr: float = 3e-4
    q_update_steps: int = 1
    target_update_steps: int = 500
    target_update_epochs: Optional[int] = None
    batch_size: int = 16
    min_experiences: int = 512
    memory_size: int = 30000
    num_episodes: int = 3000
    seed: int = 42