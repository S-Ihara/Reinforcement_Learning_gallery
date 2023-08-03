from pathlib import Path
import numpy as np
import torch

from modules.models import CNNQNet, ResnetQNet, SimpleQNet
from buffer import SimpleReplayBuffer

class DQNAgent:
    def __init__(
            self,
            observation_space: tuple,
            num_actions: int,
            gamma: float=0.99,
            max_experiences: int = 10000,
            min_experiences: int = 500,
            batch_size: int = 64,
            lr: float = 3e-4,
        ):
        """
        Args:
            gamma (float): 割引率
            max_experiences (int): リプレイバッファの最大数
            min_experiences (int): 学習を始めるのに必要な最低限のバッファのサイズ
            batch_size (int): 学習のミニバッチサイズ
            lr (float): 学習立
        TODO:
            optim周りのパラメータなど
        """
        self.observation_space = observation_space
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.max_exoeriences = max_experiences
        self.min_experiences = max(min_experiences,batch_size)
        self.gamma = gamma
        self.lr = lr

        # observation spaceの次元数でCNNかMLPかを判断したいなぁ
        # 上　これほんとぉ？
        if len(observation_space) == 3:
            # if torch.cuda.is_available():
            #     self.Q = CNNQNet(observation_space,num_actions).to("cuda")
            #     self.target_Q = CNNQNet(observation_space,num_actions).to("cuda")
            #     self.gamma = torch.tensor(gamma).to("cuda")
            # else:
            #     self.Q = CNNQNet(observation_space,num_actions)
            #     self.target_Q = CNNQNet(observation_space,num_actions)
            if torch.cuda.is_available():
                self.Q = ResnetQNet(observation_space,num_actions).to("cuda")
                self.target_Q = ResnetQNet(observation_space,num_actions).to("cuda")
                self.gamma = torch.tensor(gamma).to("cuda")
            else:
                self.Q = ResnetQNet(observation_space,num_actions)
                self.target_Q = ResnetQNet(observation_space,num_actions)
        elif len(observation_space) == 1:
            if torch.cuda.is_available():
                self.Q = SimpleQNet(observation_space,num_actions).to("cuda")
                self.target_Q = SimpleQNet(observation_space,num_actions).to("cuda")
                self.gamma = torch.tensor(gamma).to("cuda")
            else:
                self.Q = SimpleQNet(observation_space,num_actions)
                self.target_Q = SimpleQNet(observation_space,num_actions)
        else:
            raise NotImplementedError("observation space must be 1 or 3 dimentional")
        
        self.replay_buffer = SimpleReplayBuffer(
            state_shape=observation_space,
            action_shape=1,
            size=max_experiences,
        )
        self.optimizer = torch.optim.Adam(self.Q.parameters(),lr=lr)

    def get_epsilon(self,num_episode: int):
        """epsilon greedyに使うepsilonの算出
        Args:
            num_episode (int): 現在のエピソード数
        Returns:
            float: epsilon (0 <= epsilon <= 1)
        """
        return max(0.05,0.5-num_episode*0.02)
    
    def get_action(self,state: np.ndarray,epsilon: float):
        """
        Args:
            state (np.ndArray): 現在の状態
            epsilon (float): epsilon greedyに使うepsilon
        Returns
            int: action
        """
        if np.random.random() < epsilon:
            action = np.random.choice(self.num_actions)
        else:
            with torch.no_grad():
                state = torch.tensor(state,dtype=torch.float).div_(255).unsqueeze(0) # TODO ここで255で割りたくない 画像入力か状態ベクトル入力かわからないので
                action = torch.argmax(self.Q(state)).item()
        return action

    def update_networks(self):
        """
        Returns:
            float: loss
        """
        if len(self.replay_buffer) < self.min_experiences:
            return 0
    
        states,actions,rewards,next_states,dones = self.replay_buffer.get_minibatch(self.batch_size)

        states = torch.from_numpy(states) # TODO ここも255で割りたくない
        actions = torch.from_numpy(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.from_numpy(next_states)
        dones = torch.tensor(dones, dtype=torch.bool)

        if torch.cuda.is_available():
            actions = actions.to("cuda")
            states = states.to("cuda")
            rewards = rewards.to("cuda")
            dones = dones.to("cuda")
        
        current_Q_values = self.Q(states).gather(1,actions.type(torch.int64))
        next_max_q = self.target_Q(next_states).detach().max(1)[0].unsqueeze(1)
        next_Q_values = ~dones * next_max_q
        target_Q_values = rewards + (self.gamma * next_Q_values)

        loss = (target_Q_values - current_Q_values) ** 2
        loss = torch.mean(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def update_target_networks(self):
        self.target_Q.load_state_dict(self.Q.state_dict())

    def store_experience(self,state,action,reward,next_state,done):
        self.replay_buffer.store(state,action,reward,next_state,done)
    
    def save_model(self,path: Path=Path("weights")):
        """
        Args:
            path (Path): 保存先のパス
        """
        if not path.exists():
            path.mkdir()
        path = Path(path,"q.pth")
        torch.save(self.Q.state_dict(),path)
        print(f"models is saved in {path}")

    def load_model(self,path: Path=Path("weights")):
        """
        Args:
            path (Path): 保存先のパス
        """
        path = Path(path,"q.pth")
        if torch.cuda.is_available():
            self.Q.load_state_dict(torch.load(path,map_location=torch.device("cuda")))
        else:
            self.Q.load_state_dict(torch.load(path,map_location=torch.device("cpu")))
        print("model is loaded")