# from .minigrid_env import make_minigrid_env
from .classic_gym_env import make_classic_env

classic_env_list = [
    "Pendulum-v1","BipedalWalker-v3","BipedalWalkerHardcore-v3",
]

def create_env(env_name: str, **kwargs):
    if env_name in classic_env_list:
        env = make_classic_env(env_name, **kwargs)
    else:
        raise NotImplementedError(f"env_name: {env_name} is not supported")
    return env