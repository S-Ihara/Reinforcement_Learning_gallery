"""
実行ファイル
学習もテストもここから行う
"""
import argparse
from typing import Optional
from utils import Configs
from agent import DQNAgent

from envs.classic_gym_env import make_classic_env

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

    def online_train(self,num_steps: Optional[int] ,num_episodes: int = 1000):
        """訓練を行うメソッド
        Args:
            num_episodes int: 訓練するエピソード数
            num_steps int: 訓練するステップ数(Noneの場合は無視され、intの場合はnum_episodesより優先される)
        """
        env = self.env
        agent = self.agent
        total_steps = 0

        if num_steps is not None:
            num_episodes = num_steps // env.max_steps
            #TODO env にmax_stepsが必ずしもあるとは限らないのでどうにかする

        for episode in range(1,num_episodes+1):
            state,_ = env.reset()
            epsilon = agent.get_epsilon(episode)
            total_reward = 0
            step = 0
            done = False
            while not done:
                action = agent.get_action(state,epsilon)
                next_state,reward,terminated,truncated,info = env.step(action) 
                total_reward += reward
                agent.store_experience(state,action,reward,next_state,truncated)
                loss = agent.update_network()
                self.loss_history.append(loss)
                state = next_state
                step += 1
                if terminated or truncated:
                    done = True
                    total_steps += step

            agent.update_target_network()
            self.reward_history.append(total_reward)
            print(f"Episode: {episode}, Step: {step}, Total_Step; {total_steps} , Reward: {total_reward}")
        agent.save_model()
        print("Training finished")

    def test(self):
        pass


if __name__ == "__main__":
    # コマンドライン引数
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='test mode')
    args = parser.parse_args()

    config = Configs()
    env = make_classic_env(config.env_name)

    if args.test:
        # テストモード
        pass
    else:
        # 訓練モード

        pass