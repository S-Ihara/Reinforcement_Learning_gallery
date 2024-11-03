from .atari_env import make_atari_env
from .minigrid_env import make_minigrid_env

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

def create_env(env_name: str, **kwargs):
    if "MiniGrid" in env_name:
        env = make_minigrid_env(env_name, **kwargs)
    else:
        env = make_atari_env(env_name, **kwargs)
    return env