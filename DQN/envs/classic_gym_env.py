import gymnasium as gym


# LunarLander-v2

def make_classic_env(env_name: str, **kwargs):
    """
    Args:
        env_name: name of the environment
        obs_mode: 'classic' or 'rgb'
    """
    env = gym.make(env_name)
    return env
