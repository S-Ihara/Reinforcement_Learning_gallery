import gymnasium as gym

# Acrobot-v1
# CartPole-v1
# LunarLander-v2
# MountainCar-v0

def make_classic_env(env_name: str, **kwargs):
    """
    Args:
        env_name: name of the environment
        obs_mode: 'classic' or 'rgb'
    Note:
        obs_mode='rgb' is not supported yet.
    """
    env = gym.make(env_name)
    return env
