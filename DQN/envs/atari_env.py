import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing

def make_atari_env(env_name: str, gray: bool = False, size: int = 84):
    env = gym.make(env_name)
    env = AtariPreprocessing(env, screen_size=size, grayscale_obs=gray, grayscale_newaxis=True)
    return env