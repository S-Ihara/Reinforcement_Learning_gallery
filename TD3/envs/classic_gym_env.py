import gymnasium as gym

# classic
# Pendulum-v1

# box2d
# BipedalWalker-v3


def make_classic_env(env_name: str, **kwargs):
    """
    Args:
        env_name: name of the environment
        obs_mode: 'classic' or 'rgb'
    Note:
        obs_mode='rgb' is not supported yet.
    """
    if "Hardcore" in env_name:
        env = gym.make("BipedalWalker-v3",render_mode='rgb_array', hardcore=True)
    else:
        env = gym.make(env_name, render_mode='rgb_array')
    return env