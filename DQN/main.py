"""
実行ファイル
学習もテストもここから行う
# TODO
- makeenv一括でenv呼び出せるようにする
"""
import argparse
from collections import deque
from typing import Optional

import numpy as np
from gymnasium.wrappers import RecordVideo

from utils import Configs
from agent import DQNAgent
from envs import create_env

class Trainer:
    """
    訓練を制御するクラス
    """
    def __init__(self,env,agent):
        """
        Args:
            env: 自作ラッパーによる環境
            agent: エージェント
        自作クラスなのでとりあえず型タイプはしない
        """
        self.env = env
        self.agent = agent
        self.reward_history = []
        self.loss_history = []

    def online_train(self,num_steps: Optional[int]=None ,num_episodes: int = 1000, frame_stack: int = 1):
        """訓練を行うメソッド
        Args:
            num_episodes int: 訓練するエピソード数
            num_steps int: 訓練するステップ数(Noneの場合は無視され、intの場合はnum_episodesより優先される)
            frame_stack int: フレームスタック数
        """
        env = self.env
        agent = self.agent
        total_steps = 0

        if num_steps is not None:
            num_episodes = num_steps // env.max_steps
            #TODO env にmax_stepsが必ずしもあるとは限らないのでどうにかする

        for episode in range(1,num_episodes+1):
            state,_ = env.reset()
            frames = deque([state]*frame_stack,maxlen=frame_stack)
            state = np.stack(frames,axis=1).reshape(-1,*state.shape[1:3])
            epsilon = agent.get_epsilon(episode)
            total_reward = 0
            step = 0
            done = False
            while not done:
                action = agent.get_action(state,epsilon)
                next_state,reward,terminated,truncated,info = env.step(action) 
                frames.append(next_state)
                next_state = np.stack(frames,axis=1).reshape(-1,*next_state.shape[1:3])
                total_reward += reward
                agent.store_experience(state,action,reward,next_state,truncated)
                loss = agent.update_networks()
                self.loss_history.append(loss)
                state = next_state
                step += 1
                if terminated or truncated:
                    done = True
                    total_steps += step

            agent.update_target_networks()
            self.reward_history.append(total_reward)
            print(f"Episode: {episode}, Step: {step}, Reward: {total_reward}, Total_Step; {total_steps}")
        agent.save_model()
        print("Training finished")

    def test(self,num_episodes: int = 1):
        """テストを行うメソッド
        Args:
            num_episodes int: テストするエピソード数
        """
        env = self.env
        agent = self.agent
        total_steps = 0
        for episode in range(1,num_episodes+1):
            state,_ = env.reset()
            total_reward = 0
            step = 0
            done = False
            while not done:
                action = agent.get_action(state,0)
                next_state,reward,terminated,truncated,info = env.step(action) 
                total_reward += reward
                state = next_state
                step += 1
                if terminated or truncated:
                    done = True
                    total_steps += step
            print(f"episode: {episode}, step: {step}, reward: {total_reward}")


if __name__ == "__main__":
    # コマンドライン引数
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='test mode')
    args = parser.parse_args()

    config = Configs()
    #env = make_atari_env(config.env_name,size=84,gray=True)
    #env = make_minigrid_env(config.env_name)
    env = create_env(env_name=config.env_name,tile_size=16)
    obs_space = list(env.observation_space.shape)
    obs_space[0] = obs_space[0] * config.frame_stack
    obs_space = tuple(obs_space)
    num_actions = env.action_space.n

    if args.test:
        # テストモード
        env = RecordVideo(env,video_folder="video",episode_trigger=lambda x: True)
        agent = DQNAgent(
            observation_space=obs_space,
            num_actions=num_actions,
        )
        agent.load_model()
        trainer = Trainer(env,agent)
        trainer.test(num_episodes=5)
    else:
        # 訓練モード
        agent = DQNAgent(
            observation_space=obs_space,
            num_actions=num_actions,
            gamma=config.gamma,
            lr=config.lr,
            batch_size=config.batch_size,
            min_experiences=config.min_experiences,
            max_experiences=config.memory_size,
        )
        trainer = Trainer(env,agent)
        trainer.online_train(num_episodes=config.num_episodes)