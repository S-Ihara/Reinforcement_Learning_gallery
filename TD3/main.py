"""
実行ファイル
学習もテストもここから行う
"""
import argparse
from collections import deque
from typing import Optional
from pathlib import Path
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ray import tune
from gymnasium.wrappers import RecordVideo

import configs
from configs import DefaultConfigs
from utils import set_random_seed
from agent import DDPGAgent
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
        """
        self.env = env
        self.agent = agent
        self.reward_history = []
        self.loss_history = []

    def online_train(
            self,
            num_episodes: int = 1000, 
            frame_stack: int = 1,
            update_steps: int = 1,
            reward_clip: bool = False,
            save_model: bool = False,
            noise: float = 0.3,
        ):
        """訓練を行うメソッド
        Args:
            num_episodes int: 訓練するエピソード数
            frame_stack int: フレームスタック数
            update_steps int: モデルの更新ステップ間隔
            reward_clip bool: 報酬のクリッピング(-1 ~ 1)を行うかどうか
            save_model bool: モデルの保存を行うかどうか
            noise float: 行動値のランダム化の幅
        """
        env = self.env
        agent = self.agent
        total_steps = 0

        for episode in range(1,num_episodes+1):
            state,_ = env.reset()
            frames = deque([state]*frame_stack,maxlen=frame_stack)
            state = np.stack(frames,axis=1).reshape(-1,*state.shape[1:3])
            episode_losses = []
            critic_losses = []
            actor_losses = []
            total_reward = 0
            step = 0
            done = False

            while not done:
                action = agent.get_actions(state,noise=noise)

                next_state,reward,terminated,truncated,info = env.step(action) 
                if reward_clip:
                    reward = np.clip(reward,-1,1)

                frames.append(next_state)
                next_state = np.stack(frames,axis=1).reshape(-1,*next_state.shape[1:3])
                total_reward += reward
                agent.store_experience(state,action,reward,next_state,terminated)
                if total_steps % update_steps == 0:
                    loss_dict = agent.update_networks()
                    loss = loss_dict["critic_loss"] + loss_dict["actor_loss"]
                    critic_losses.append(loss_dict["critic_loss"])
                    actor_losses.append(loss_dict["actor_loss"])
                    self.loss_history.append(loss)
                    episode_losses.append(loss)

                    agent.update_target()
                state = next_state
                step += 1
                if terminated or truncated:
                    done = True
                    total_steps += step

            self.reward_history.append(total_reward)
            if episode_losses == []:
                episode_losses = 0
                actor_losses = 0
                critic_losses = 0
            else:
                episode_losses = np.mean(episode_losses)
                actor_losses = np.mean(actor_losses)
                critic_losses = np.mean(critic_losses)
            print(f"Episode: {episode}, Total_Step: {total_steps}, Step: {step}, Reward: {total_reward}, actor_loss: {actor_losses}, critic_loss: {critic_losses}")
        if save_model:
            agent.save_model()
        print("Training finished")

        return {"mean_reward": np.mean(self.reward_history[-100:])}

    
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
        

    def test(self,num_episodes: int = 1, frame_stack: int = 1):
        """テストを行うメソッド
        Args:
            num_episodes int: テストするエピソード数
            frame_stack int: フレームスタック数
        """
        env = self.env
        agent = self.agent
        total_steps = 0
        for episode in range(1,num_episodes+1):
            state,_ = env.reset()
            frames = deque([state]*frame_stack,maxlen=frame_stack)
            state = np.stack(frames,axis=1).reshape(-1,*state.shape[1:3])
            total_reward = 0
            step = 0
            done = False
            while not done:
                action = agent.get_actions(state,0.)
                next_state,reward,terminated,truncated,info = env.step(action) 
                frames.append(next_state)
                next_state = np.stack(frames,axis=1).reshape(-1,*next_state.shape[1:3])
                total_reward += reward
                state = next_state
                step += 1
                if terminated or truncated:
                    done = True
                    total_steps += step
            print(f"episode: {episode}, step: {step}, reward: {total_reward}")
        env.close()


def objective(config,options):
    """ray tune用の目的関数"""
    obs_space = options["obs_space"]
    num_actions = options["num_actions"]
    action_range = options["action_range"]
    env = options["env"]
    reward_clip = options["reward_clip"]

    batch_size = config["batch_size"]
    min_experiences = config["min_experiences"]
    memory_size = config["memory_size"]
    lr = config["lr"]
    gamma = config["gamma"]
    tau = config["tau"]
    update_steps = config["update_steps"]
    hidden_dim = config["hidden_dim"]
    noise = config["noise"]
    #critic_hidden_dim = config["critic_hidden_dim"]
    #actor_hidden_dim = config["actor_hidden_dim"]
    activation = config["activation"]

    agent = DDPGAgent(
        observation_space=obs_space,
        num_actions=num_actions,
        gamma=gamma,
        tau=tau,
        lr=lr,
        batch_size=batch_size,
        min_experiences=min_experiences,
        max_experiences=memory_size,
        action_range=action_range,
        hidden_dim=hidden_dim,
        #critic_hidden_dim=critic_hidden_dim,
        #actor_hidden_dim=actor_hidden_dim,
        activation=activation,
    )
    trainer = Trainer(env,agent)
    returns = trainer.online_train(
        num_episodes=300,
        update_steps=update_steps,
        reward_clip=reward_clip,
        noise=noise,
    )

    return returns 

if __name__ == "__main__":
    # コマンドライン引数
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument('--tuning', action='store_true', help='tuning mode')
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

    # model parameters
    hidden_dim = getattr(config,"hidden_dim",64)
    activation = getattr(config,"activation","ReLU")

    # ddpg paramters
    noise = getattr(config,"noise",0.3)

    env = create_env(
        env_name=config.env_name,tile_size=tile_size,size=size,gray=gray
    )
    obs_space = list(env.observation_space.shape)
    obs_space[0] = obs_space[0] * config.frame_stack
    obs_space = tuple(obs_space)
    print(f"action_space: {env.action_space}")
    num_actions = env.action_space.shape[0]
    action_range = env.action_space.high[0]

    if args.tuning:
        # ray tune
        search_space = {
            "gamma": tune.choice([0.98,0.95,0.99]),
            "lr": tune.choice([1e-3,1e-4]),
            "tau": tune.choice([0.001,0.003,0.01]),
            "batch_size": tune.choice([8,16,32,64]),
            "min_experiences": tune.choice([2000]),
            "memory_size": tune.choice([20000,50000]),
            "update_steps": tune.choice([1,2,4]),
            "hidden_dim": tune.choice([64]),
            #"critic_hidden_dim": tune.choice([16,32,64,128]),
            #"actor_hidden_dim": tune.choice([16,32,64,128]),
            "activation": tune.choice(["ReLU"]),
            "noise": tune.choice([0.1,0.3,0.5,0.7]),
        }
        options = {
            "obs_space": obs_space,
            "num_actions": num_actions,
            "action_range": action_range,
            "env": env,
            "reward_clip": reward_clip,
        }
        search_alg = tune.search.basic_variant.BasicVariantGenerator()
        tuner = tune.Tuner(
            tune.with_resources(
                partial(objective, options=options),
                resources={'cpu': 1, 'gpu': 0},
            ),
            tune_config = tune.TuneConfig(
                search_alg=search_alg,
                num_samples=500,
            ),
            param_space=search_space,
        )
        results = tuner.fit()
        print(results.get_best_result(metric="mean_reward", mode="max").config)

        exit()

    if args.test:
        # テストモード
        env = RecordVideo(env,video_folder="video",episode_trigger=lambda x: True)
        agent = DDPGAgent(
            observation_space=obs_space,
            num_actions=num_actions,
            action_range=action_range,
            hidden_dim=hidden_dim,
            activation=activation,
        )
        agent.load_model()
        trainer = Trainer(env,agent)
        trainer.test(num_episodes=5)
    else:
        # 訓練モード
        agent = DDPGAgent(
            observation_space=obs_space,
            num_actions=num_actions,
            gamma=config.gamma,
            tau=config.tau,
            lr=config.lr,
            batch_size=config.batch_size,
            min_experiences=config.min_experiences,
            max_experiences=config.memory_size,
            action_range=action_range,
            hidden_dim=hidden_dim,
            activation=activation,
        )

        trainer = Trainer(env,agent)
        trainer.online_train(
            num_episodes=config.num_episodes,
            update_steps=config.update_steps,
            reward_clip=reward_clip,
            save_model=True,
            noise=noise,
        )
        trainer.plot_history()