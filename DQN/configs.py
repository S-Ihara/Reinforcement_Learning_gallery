from dataclasses import dataclass

@dataclass
class DefaultConfigs:
    """ハイパーパラメータ
    """
    env_name: str = "Breakout-v4"
    frame_stack: int = 1
    gray: bool = True
    gamma: float = 0.96
    lr: float = 4e-4
    q_update_steps: int = 2
    target_update_steps: int = 500
    target_update_epochs: int|None = None
    batch_size: int = 128
    min_experiences: int = 512
    memory_size: int = 100000
    num_episodes: int = 3000
    seed: int = 42

@dataclass
class CartPole:
    """CartPole-v1
    """
    env_name: str = "CartPole-v1"
    frame_stack: int = 1
    gamma: float = 0.96
    lr: float = 3e-4
    q_update_steps: int = 1
    target_update_steps: int = 500
    target_update_epochs: int|None = None
    batch_size: int = 256
    min_experiences: int = 512
    memory_size: int = 100000
    num_episodes: int = 3000
    seed: int = 42

@dataclass
class Acrobot:
    env_name: str = "Acrobot-v1"
    frame_stack: int = 1
    gamma: float = 0.95
    lr: float = 4e-4
    q_update_steps: int = 1
    target_update_steps: int = 3000
    target_update_epochs: int|None = None
    batch_size: int = 64
    min_experiences: int = 1024
    memory_size: int = 100000
    num_episodes: int = 3000
    seed: int = 42
    reward_clip: bool = True

@dataclass
class Atari:
    env_name: str = "SpaceInvaders-v4"
    frame_stack: int = 3
    gray: bool = True
    gamma: float = 0.99
    lr: float = 3e-4
    q_update_steps: int = 1
    target_update_steps: int = 1000
    target_update_epochs: int|None = None
    batch_size: int = 512
    min_experiences: int = 2048
    memory_size: int = 200000
    num_episodes: int = 3000
    seed: int = 42
    reward_clip: bool = True

@dataclass
class Breakout:
    env_name: str = "Breakout-v4"
    frame_stack: int = 3
    gray: bool = True
    gamma: float = 0.99
    lr: float = 3e-4
    q_update_steps: int = 1
    target_update_steps: int = 1000
    target_update_epochs: int|None = None
    batch_size: int = 512
    min_experiences: int = 2048
    memory_size: int = 300000
    num_episodes: int = 3000
    seed: int = 42

@dataclass
class MiniGridEmpty:
    env_name: str = "MiniGrid-Empty-8x8-v0" # "MiniGrid-Empty-Random-6x6-v0"
    tile_size: int = 8
    frame_stack: int = 1
    gamma: float = 0.95
    lr: float = 3e-4
    q_update_steps: int = 1
    target_update_steps: int = 500
    target_update_epochs: int|None = None
    batch_size: int = 128
    min_experiences: int = 512
    memory_size: int = 100000
    num_episodes: int = 5000
    seed: int = 42