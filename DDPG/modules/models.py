import torch
import torch.nn as nn

from modules.feature_extractor import SimpleCNN

devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorNetwork(nn.Module):
    def __init__(self,observation_space: tuple, num_actions: int, **kwargs):
        """
        Args:
            observation_space (tuple): 状態空間のtensor shape
            num_actions (int): 行動空間の次元数
            # 以下kwargs
            action_range (float): それぞれの行動の範囲 (-1 ~ 1)*action_range
            hidden_dim (int): 隠れ層の次元数
            activation (str): 活性化関数
        """
        super(ActorNetwork,self).__init__()
        hidden_dim = kwargs.get("hidden_dim",64)
        act_func = getattr(nn,kwargs.get("activation","ReLU"))()

        if len(observation_space) == 3:
            self.feature_extractor = SimpleCNN(observation_space)
            feature_dim = self.feature_extractor.output_feature_dim
        elif len(observation_space) == 1:
            self.feature_extractor = nn.Identity()
            feature_dim = observation_space[0]
        else:
            raise NotImplementedError(f"observation space: {observation_space} is not supported")

        self.action_head = nn.Sequential(
            nn.Linear(feature_dim,hidden_dim),
            act_func,
            nn.Linear(hidden_dim,hidden_dim),
            act_func,
            nn.Linear(hidden_dim,num_actions),
            nn.Tanh(),
        )
        self.action_range = kwargs.get("action_range",1)

    def forward(self,s):
        x = self.feature_extractor(s)
        x = self.action_head(x)
        x = x*self.action_range

        return x 

class CriticeNetwork(nn.Module):
    def __init__(self,observation_space: tuple, num_actions: int, **kwargs):
        """
        Args:
            observation_space (tuple): 状態空間のtensor shape
            num_actions (int): 行動空間の次元数
            # 以下kwargs
            hidden_dim (int): 隠れ層の次元数
            activation (str): 活性化関数
        """
        super(CriticeNetwork,self).__init__()
        hidden_dim = kwargs.get("hidden_dim",64)
        act_func = getattr(nn,kwargs.get("activation","ReLU"))()

        if len(observation_space) == 3:
            self.feature_extractor = SimpleCNN(observation_space)
            feature_dim = self.feature_extractor.output_feature_dim
        elif len(observation_space) == 1:
            self.feature_extractor = nn.Identity()
            feature_dim = observation_space[0]
        else:
            raise NotImplementedError(f"observation space: {observation_space} is not supported")

        self.value_head = nn.Sequential(
            nn.Linear(feature_dim+num_actions,hidden_dim),
            act_func,
            nn.Linear(hidden_dim,hidden_dim),
            act_func,
            nn.Linear(hidden_dim,1),
        )

    def forward(self,s,a):
        s = self.feature_extractor(s)
        x = torch.cat([s,a],axis=1)
        x = self.value_head(x)
        
        return x