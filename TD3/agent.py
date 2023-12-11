from pathlib import Path
import numpy as np
import torch

from modules.models import ActorNetwork, CriticeNetwork
from replay_buffer import SimpleReplayBuffer

class DDPGAgent:
    def __init__(
            self,
            observation_space: tuple,
            num_actions: int,
            gamma: float=0.99,
            tau: float=0.005,
            max_experiences: int = 10000,
            min_experiences: int = 500,
            batch_size: int = 64,
            lr: float = 3e-4,
            action_range: float = 1,
            **kwargs
        ):
        """
        Args:
            observation_space (tuple): 状態空間の次元数
            num_actions (int): 行動空間の次元数
            gamma (float): 割引率
            tau (float): soft-target updateの係数
            max_experiences (int): リプレイバッファの最大数
            min_experiences (int): 学習を始めるのに必要な最低限のバッファのサイズ
            batch_size (int): 学習のミニバッチサイズ
            lr (float): 学習率
            action_range (float): 行動値の範囲 (-1 ~ 1)*action_range
            **kwargs:
                hidden_dim (int): 隠れ層の次元数 (共通)
                critic_hidden_dim (int): criticの隠れ層の次元数
                actor_hidden_dim (int): actorの隠れ層の次元数 # これらが設定されてたら優先される
                activation (str): 活性化関数
                num_critics (int): criticの数
        TODO:
            optim周りのパラメータなど
        """
        self.observation_space = observation_space
        self.num_actions = num_actions
        self.action_range = action_range
        self.batch_size = batch_size
        self.max_exoeriences = max_experiences
        self.min_experiences = max(min_experiences,batch_size)
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.update_actor_flag = False
        
        num_critics = kwargs.get("num_critics",2)
        hidden_dim = kwargs.get("hidden_dim",64)
        if "critic_hidden_dim" in kwargs:
            critic_hidden_dim = kwargs.get("critic_hidden_dim")
            actor_hidden_dim = kwargs.get("actor_hidden_dim")
        else:
            critic_hidden_dim = hidden_dim
            actor_hidden_dim = hidden_dim
        activation = kwargs.get("activation","ReLU")

        self.critic = CriticeNetwork(observation_space,num_actions,hidden_dim=critic_hidden_dim,activation=activation,num_critics=num_critics)
        self.target_critic = CriticeNetwork(observation_space,num_actions,hidden_dim=critic_hidden_dim,activation=activation,num_critics=num_critics)
        self.actor = ActorNetwork(observation_space,num_actions,action_range=action_range,hidden_dim=actor_hidden_dim,activation=activation)
        self.target_actor = ActorNetwork(observation_space,num_actions,action_range=action_range,hidden_dim=actor_hidden_dim,activation=activation)
        if torch.cuda.is_available():
            self.critic = self.critic.to("cuda")
            self.target_critic = self.target_critic.to("cuda")
            self.actor = self.actor.to("cuda")
            self.target_actor = self.target_actor.to("cuda")

        self.replay_buffer = SimpleReplayBuffer(
            state_shape=observation_space,
            action_shape=num_actions,
            size=max_experiences,
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=lr)
        #self.loss_fn = torch.nn.MSELoss()
        self.loss_fn = torch.nn.HuberLoss()

    @torch.no_grad()
    def get_actions(self, state ,noise: float=0.):
        """
        Args:
            state (np.ndarray): 現在の状態
            noise (float): 行動に混ぜる平均0の正規分布ノイズの標準偏偏差
        """
        if torch.cuda.is_available():
            state = torch.tensor(state,dtype=torch.float,device="cuda").unsqueeze(0)
        else:
            state = torch.tensor(state,dtype=torch.float).unsqueeze(0)
        action = self.actor(state)[0]

        if noise != 0.:
            if torch.cuda.is_available():
                action += torch.normal(0,noise,size=action.shape,device="cuda")
            else:
                action += torch.normal(0,noise,size=action.shape)
            action = torch.clamp(action,-self.action_range,self.action_range) # actionのクリッピング 
        
        return action.detach().cpu().numpy()

    def update_networks(self):
        if len(self.replay_buffer) < self.min_experiences:
            return {"critic_loss": 0,"actor_loss": 0}
        
        states,actions,rewards,next_states,dones = self.replay_buffer.get_minibatch(self.batch_size)

        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.from_numpy(next_states)
        dones = torch.tensor(dones, dtype=torch.bool)

        if torch.cuda.is_available():
            actions = actions.to("cuda")
            states = states.to("cuda")
            rewards = rewards.to("cuda")
            dones = dones.to("cuda")
            next_states = next_states.to("cuda")
        
        with torch.no_grad():
            # clip noise
            clipped_noise = torch.clamp(torch.normal(0,0.2,size=actions.shape,device=actions.device),-0.5,0.5)
            next_actions = self.target_actor(next_states) + clipped_noise
            next_qvalues_list = self.target_critic(next_states,next_actions)
            next_qvalues = torch.min(*next_qvalues_list)
        #import pdb; pdb.set_trace()
        
        # update critic network
        next_qvalues = ~dones * next_qvalues
        Q_targets = rewards + self.gamma*next_qvalues
        #Q_values = self.critic(states,actions)

        Q_values_list = self.critic(states,actions)
        Q_losses = []
        for Q_values in Q_values_list:
            Q_losses.append(self.loss_fn(Q_targets,Q_values))
        #critic_loss = torch.mean(torch.stack(Q_losses))
        critic_loss = torch.sum(torch.stack(Q_losses))

        #critic_loss = self.loss_fn(Q_targets,Q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor network
        if self.update_actor_flag:
            Q = self.critic(states,self.actor(states))[0] # これは最小値とかではなく0番目のcriticのQ値とかでいいらしい ほんとか？
            J = -1 * torch.mean(Q)
            #J = -1 * torch.mean(self.critic(states,self.actor(states)))

            self.actor_optimizer.zero_grad()
            J.backward()
            self.actor_optimizer.step()
            actor_loss = J.detach().cpu().item()
        else:
            actor_loss = 0
        self.update_actor_flag = not self.update_actor_flag

        return {"critic_loss": critic_loss.detach().cpu().item(),"actor_loss": actor_loss}

    def update_target(self):
        """soft-target update
        """
        # actor
        for target_actor_param, actor_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_actor_param.data.copy_(
                target_actor_param.data * (1.0 - self.tau) + actor_param.data * self.tau
            )

        # critic
        for target_critic_param, critic_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_critic_param.data.copy_(
                target_critic_param.data * (1.0 - self.tau) + critic_param.data * self.tau
            )

    def store_experience(self,state,action,reward,next_state,done):
        self.replay_buffer.store(state,action,reward,next_state,done)
    
    def save_model(self,path: Path=Path("weights")):
        """
        Args:
            path (Path): 保存先のパス
        """
        if not path.exists():
            path.mkdir()
        actor_path = Path(path,"actor.pth")
        critic_path = Path(path,"critic.pth")
        torch.save(self.critic.state_dict(),critic_path)
        torch.save(self.actor.state_dict(),actor_path)
        print(f"models is saved in {path}")

    def load_model(self,path: Path=Path("weights")):
        """
        Args:
            path (Path): 保存先のパス
        """
        actor_path = Path(path,"actor.pth")
        critic_path = Path(path,"critic.pth")
        if torch.cuda.is_available():
            self.critic.load_state_dict(torch.load(critic_path,map_location=torch.device("cuda")))
            self.actor.load_state_dict(torch.load(actor_path,map_location=torch.device("cuda")))
        else:
            self.critic.load_state_dict(torch.load(critic_path,map_location=torch.device("cpu")))
            self.actor.load_state_dict(torch.load(actor_path,map_location=torch.device("cpu")))

        print("model is loaded")