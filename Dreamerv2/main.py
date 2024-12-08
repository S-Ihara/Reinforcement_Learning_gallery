"""
実行ファイル
学習もテストもここで行う
"""
from pathlib import Path
from copy import deepcopy
import gc 

import matplotlib.pyplot as plt
import torch 
from gymnasium.wrappers import RecordVideo

from agent import DreamerAgent
from buffer import ReplayBuffer
from envs import create_env
from envs.env_utils import check_env_stats


# 仮のconfig
INIT_EPISODE = 50
COLLECT_INTERVALS = 5

SEQ_LEN = 50
BATCH_SIZE = 512
TRAIN_STEPS = 10000
MEMORY_SIZE = 100000

class Trainer:
    def __init__(self, env, agent):
        """
        args:
            env (gymnasium.Env): 環境
            agent (DreamerAgent): エージェント
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.agent = agent
        self.train_steps = TRAIN_STEPS
        self.collect_intervals = COLLECT_INTERVALS

        self.buffer = ReplayBuffer(capacity=MEMORY_SIZE, state_shape=env.observation_space.shape, action_size=env.action_space.n, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)

        self.log_dir = Path("./logs")
        self.log_dir.mkdir(exist_ok=True)
    
    def train(self):
        """学習を行う
        """
        print("start training")
        agent = self.agent

        # initial collection of experience
        for _ in range(INIT_EPISODE):
            self._collect_experience()
        
        # training loop
        global_step = 0 

        while True:
            print("#"*20)
            print(f"global step: {global_step}")
            self._update_networks()
            global_step += 1

            self._collect_experience(policy="agent")
            if global_step == self.train_steps:
                break 

            if global_step % 10 == 0:
                self.testplay(video=True)

        
        # モデルの保存
        agent.save(str(self.log_dir / "agent.pth"))

        print("training finished")

    def _collect_experience(self, policy="random"):
        """
        1エピソード分の経験を収集する
        args:
            policy (str): 行動選択方策 "random" or "agent"
        """
        env = self.env
        state, info = env.reset()

        while True:
            if policy == "random":
                action = env.action_space.sample()
            elif policy == "agent":
                action = self.agent.get_action(state=state)

            next_state, reward, terminated, truncated, info = env.step(action)
            self.buffer.add(state, action, reward, terminated)

            state = next_state

            if terminated or truncated:
                break 
        
    def _update_networks(self):
        """ネットワークを更新する
        """
        print("update networks")
        for i in range(self.collect_intervals):
            states, actions, rewards, terminals = self.buffer.sample() # shape=(Length, Batch, *Shape)

            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            terminals = torch.tensor(terminals, dtype=torch.float32).to(self.device)

            world_model_loss = self.agent.train_worldmodel_minibatch(states, actions, rewards, terminals)
            print(f'total world model loss: {world_model_loss["loss"]:.4f}')
            print(f'reconstruction loss: {world_model_loss["img_reconstruction_loss"]: .4f}')
            print(f'reward loss: {world_model_loss["reward_loss"]:.4f}')
            print(f'discount loss: {world_model_loss["discount_loss"]: .4f}')
            print(f'kl loss: {world_model_loss["kl_loss"]: .4f}')

            sampled_z = world_model_loss["sampled_z"] # torch.cat(world_model_loss["sampled_z"])
            sampled_h = world_model_loss["sampled_h"] # torch.cat(world_model_loss["sampled_h"])
            actor_critic_loss = self.agent.train_actor_critic(sampled_z, sampled_h)
            print(f'value loss: {actor_critic_loss["value_loss"]: .4f}')
            print(f'policy loss: {actor_critic_loss["policy_loss"]: .4f}')

            print()

    def testplay(self, video=False, reconstruciton=True):
        """テストプレイを行う
        args:
            video (bool): ビデオ保存するかどうか
            reconstruciton (bool): 再構成画像動画を保存するかどうか
        """
        print("test play")
        if video:
            save_dir = str(self.log_dir / "video")
            env = deepcopy(self.env)
            env = RecordVideo(env,video_folder=save_dir ,episode_trigger=lambda x: True)
        else:
            env = self.env

        if reconstruciton:
            (self.log_dir / "reconstruction").mkdir(exist_ok=True)

        state, info = env.reset()
        time_step = 0
        while True:
            if reconstruciton:  
                action = self.agent.get_action(state, return_reconstruction=True)
                reconstruct_image = action["reconstruction"]
                action = action["action"]
            else:
                action = self.agent.get_action(state=state)

            next_state, reward, terminated, truncated, info = env.step(action)
            self.buffer.add(state, action, reward, terminated)

            state = next_state
            time_step += 1

            if reconstruciton:
                fig, ax = plt.subplots()
                ax.imshow(reconstruct_image.squeeze(0).permute(1,2,0).cpu().numpy(), cmap="gray")
                ax.axis("off")
                fig.savefig(self.log_dir / "reconstruction" / f"{time_step}.png")

                del fig, ax
                gc.collect()

            if terminated or truncated:
                break 

if __name__ == "__main__":
    env = create_env("ALE/Boxing-v5", size=64) # 一旦Breakoutで動かす
    check_env_stats(env)

    agent = DreamerAgent(latent_dim=32, hidden_dim=128, n_atoms=8, action_size=env.action_space.n, img_channels=env.observation_space.shape[0])  

    trainer = Trainer(env=env, agent=agent)
    trainer.train()