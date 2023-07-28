import gymnasium as gym

def make_classic_env(env_name: str):
    env = gym.make(env_name)
    return env
