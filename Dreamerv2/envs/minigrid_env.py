"""minigird環境
Note:
    requirements
    pip install minigrid

Env_list:
    - MiniGrid-Empty-Random-5x5-v0
    - MiniGrid-Empty-Random-6x6-v0
    - MiniGrid-Empty-16x16-v0
"""
import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper
from minigrid.wrappers import RGBImgObsWrapper

def make_minigrid_env(env_name: str, **kwargs):
    """
    Args:
        env_name: name of the environment
        tile_size: size of one of tile
    """
    tile_size = kwargs.get("tile_size", 8)
    env = gym.make(env_name,render_mode='rgb_array')
    env = RGBImgObsWrapper(env,tile_size=tile_size)
    env = OnlyImageReturnWrapper(env)
    return env

class OnlyImageReturnWrapper(ObservationWrapper):
    """
    returnを画像のみにする
    """
    def __init__(self,env):
        super().__init__(env)
        self.observation_space = self.observation_space["image"]
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.observation_space.shape[2],self.observation_space.shape[0],self.observation_space.shape[1]),
            dtype=np.float32,
        )
        self.observation_space.high = self.observation_space.high.astype(np.float32) / 255.

    
    def observation(self,observation):
        observation = observation["image"]
        observation = observation.astype(np.float32) / 255.0
        return observation.transpose(2,0,1)
        return observation

if __name__ == "__main__":
    # テスト
    env = make_minigrid_env('MiniGrid-Empty-16x16-v0', tile_size=4)
    o,_ = env.reset()
    print(o)
    print(env.observation_space)
    print(env.action_space)
    render = env.render()
    print(render.shape)
