import numpy as np
from torchvision.transforms import v2 
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.wrappers import AtariPreprocessing
import ale_py

gym.register_envs(ale_py)


def make_atari_env(env_name: str , **kwargs):
    """
    Args:
        env_name: name of the environment
        size: size of the screen
        gray: whether to convert the screen to grayscale
    """
    size = kwargs.get("size", 84)
    gray = kwargs.get("gray", True)

    env = gym.make(env_name,render_mode='rgb_array')
    env = AtariPreprocessing(env,frame_skip=1 ,screen_size=size, grayscale_obs=gray, grayscale_newaxis=gray)
    env = TorchImgshapeWrapper(env)
    return env

class TorchImgshapeWrapper(ObservationWrapper):
    """
    gymの環境の出力する画像のshapeをPytorchの形式に変換する
    ついでにrangeを[0,1]に変換する
    """
    def __init__(self,env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(env.observation_space.shape[2],env.observation_space.shape[0],env.observation_space.shape[1]),
            dtype=np.float32,
        )

        # self.transform = v2.Compose([
        #     v2.ToPILImage(),
        #     v2.Resize((84,84)),
        #     v2.ToTensor()
        # ])
    
    def observation(self,observation):
        #observation = observation.astype('float32') / 255.0
        return observation.transpose(2,0,1)