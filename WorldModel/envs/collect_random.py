"""
ランダム行動で環境のデータを収集

TODO:
    - rayで並列化できるようにする
"""
from typing import Union
from pathlib import Path 

import numpy as np 
import gymnasium as gym

def generate_data(num_episodes: int, data_path: Union[str,Path], env_name: str = "CarRacing-v2"):
    """
    Args:
        num_episodes (int): 環境を何回実行するか
        data_path (Union[str,Path]): データの保存先
        env_name (str): 環境名
    """
    data_path = Path(data_path)
    data_path.mkdir(exist_ok=True,parents=True)
    env = gym.make(env_name)

    for e in range(num_episodes):
        obs,info = env.reset()

        s_rollout = []
        a_rollout = []
        r_rollout = []
        d_rollout = []

        t=0
        while True:
            action = env.action_space.sample()
            done = False
            obs,reward,terminated,truncated,info = env.step(action)

            # if terminated or truncated:
            #     done = True
            if terminated:
                done = True

            s_rollout.append(obs)
            a_rollout.append(action)
            r_rollout.append(reward)
            d_rollout.append(done)

            t+=1
            if terminated or truncated:
                print(f"episode {e:04d} finished at t={t}")
                np.savez(data_path / f"episode_{e:04d}.npz",
                         observations=np.array(s_rollout),
                         actions=np.array(a_rollout),
                         rewards=np.array(r_rollout),
                         dones=np.array(d_rollout),
                        )
                break

if __name__ == "__main__":
    """test"""
    generate_data(10,"./data")

