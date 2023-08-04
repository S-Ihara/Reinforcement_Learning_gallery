"""
実行ファイル
学習もテストもここから行う
# TODO
- makeenv一括でenv呼び出せるようにする
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
            env: 自作ラッパーによる環境
            agent: エージェント
        自作クラスなのでとりあえず型タイプはしない
        """
        self.env = env
        self.agent = agent
        self.reward_history = []
        self.loss_history = []

    def online_train(
            self,
            num_steps: Optional[int]=None,
            num_episodes: int = 1000, 
            frame_stack: int = 1,
            q_update_steps: int = 1,
            target_update_steps: int = 1000,
            target_update_epochs: Optional[int] = None,
        ):
        """訓練を行うメソッド
        Args:
            num_episodes int: 訓練するエピソード数
            num_steps int: 訓練するステップ数(Noneの場合は無視され、intの場合はnum_episodesより優先される)
            frame_stack int: フレームスタック数
            q_update_steps int: Q関数の更新ステップ間隔数
            target_update_steps int: ターゲットネットワークの更新ステップ間隔数
            target_update_epochs int: ターゲットネットワークの更新エポック数(Noneの場合は無視されるが、intの場合はtarget_update_stepsより優先される)
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
            episode_losses = []
            total_reward = 0
            step = 0
            done = False

            while not done:
                action = agent.get_action(state,epsilon)
                next_state,reward,terminated,truncated,info = env.step(action) 
                #print("debug!")
                #print(f"state: {state}, reward: {reward}, next_state: {next_state}, terminated: {terminated}, truncated: {truncated}")
                #if terminated: # CartPole
                #    reward = -1
                frames.append(next_state)
                next_state = np.stack(frames,axis=1).reshape(-1,*next_state.shape[1:3])
                total_reward += reward
                agent.store_experience(state,action,reward,next_state,terminated)
                if total_steps % q_update_steps == 0:
                    loss = agent.update_networks()
                    self.loss_history.append(loss)
                    episode_losses.append(loss)
                if (target_update_epochs is None) and (total_steps % target_update_steps) == 0:
                    agent.update_target_networks()
                state = next_state
                step += 1
                if terminated or truncated:
                    done = True
                    total_steps += step

            if (target_update_epochs is not None) and (episode % target_update_epochs == 0):
                agent.update_target_networks()
            self.reward_history.append(total_reward)
            if episode_losses == []:
                episode_losses = 0
            else:
                episode_losses = np.mean(episode_losses)
            print(f"Episode: {episode}, Step: {step}, Reward: {total_reward}, Total_Step: {total_steps}, loss_avg: {episode_losses}")
        agent.save_model()
        print("Training finished")
    
    def plot_history(self,save_path: Path = Path("./history")):
        """報酬や損失の履歴をプロットするメソッド
        Args:
            save_path Path: 保存先のパス
        """
        if not save_path.exists():
            save_path.mkdir(parents=True)
        sns.set_style("darkgrid")
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        ax.plot(self.reward_history)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        plt.savefig(save_path/"reward_history.png")

        fig,ax = plt.subplots(1,1,figsize=(10,10))
        ax.plot(self.loss_history)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        plt.savefig(save_path/"loss_history.png")
        

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


    tile_size = getattr(config,"tile_size",8)
    size = getattr(config,"size",84)
    gray = getattr(config,"gray",False)
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
        )
        trainer.plot_history()