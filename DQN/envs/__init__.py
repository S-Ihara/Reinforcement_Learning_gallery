from .atari_env import make_atari_env
from .minigrid_env import make_minigrid_env
from .classic_gym_env import make_classic_env

classic_env_list = [
    "Acrobot-v1", "CartPole-v1", "LunarLander-v2", "MountainCar-v0"
]

def create_env(env_name: str, **kwargs):
    if "MiniGrid" in env_name:
        env = make_minigrid_env(env_name, **kwargs)
    elif env_name in classic_env_list:
        env = make_classic_env(env_name, **kwargs)
    else:
        try:
            env = make_atari_env(env_name, **kwargs)
        except Exception as e:
            raise e
            #raise ValueError(f"env_name: {env_name} is not supported")
    return env