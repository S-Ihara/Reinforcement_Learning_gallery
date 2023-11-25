"""
実行ファイル
学習もテストもここから行う
"""
import argparse
from collections import deque
from typing import Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gymnasium.wrappers import RecordVideo

import configs
from configs import DefaultConfigs
from utils import set_random_seed
from agent import DQNAgent
from envs import create_env

class Trainer:
    """
    訓練を制御するクラス
    """
    def __init__(self,env,agent):
        """
        Args:
            env: 環境
            agent: エージェント
        """
        self.env = env
        self.agent = agent
        self.reward_history = []
        self.loss_history = []

        self.buffer = None

    def ofline_train(self, num_steps: int = 10000, batch_size: int = 32):
        """
        Args:
            num_steps int: 学習ステップ数
            batch_size int: ミニバッチサイズ
        """
        train_losses = []

        for _ in range(num_steps):
            states, actions, rewards, dones, rtg, timesteps = self.buffer.get_minibatch(batch_size)
    
    # def plot_history(self,save_path: Path = Path("./history")):
    #     """報酬や損失の履歴をプロットするメソッド
    #     Args:
    #         save_path Path: 保存先のパス
    #     """
    #     if not save_path.exists():
    #         save_path.mkdir(parents=True)
    #     sns.set_style("darkgrid")
    #     fig,ax = plt.subplots(1,1,figsize=(10,10))
    #     ax.plot(self.reward_history)
    #     ax.set_xlabel("Episode")
    #     ax.set_ylabel("Reward")
    #     plt.savefig(save_path/"reward_history.png")

    #     fig,ax = plt.subplots(1,1,figsize=(10,10))
    #     ax.plot(self.loss_history)
    #     ax.set_xlabel("Step")
    #     ax.set_ylabel("Loss")
    #     plt.savefig(save_path/"loss_history.png")
        

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
    parser.add_argument('-c','--config', type=str, default='DefaultConfigs', help='config file')
    args = parser.parse_args()

    config = getattr(configs,args.config)()
    print(f"current config: {config}")
    set_random_seed(config.seed)

    # optional parameters
    tile_size = getattr(config,"tile_size",8)
    size = getattr(config,"size",84)
    gray = getattr(config,"gray",False)
    reward_clip = getattr(config,"reward_clip",False)

    env = create_env(
        env_name=config.env_name,tile_size=tile_size,size=size,gray=gray
    )
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
        trainer.online_train(
            num_episodes=config.num_episodes,
            q_update_steps=config.q_update_steps,
            target_update_steps=config.target_update_steps,
            target_update_epochs=config.target_update_epochs,
            reward_clip=reward_clip,
        )
        trainer.plot_history()