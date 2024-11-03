import torch 
import torch.nn as nn
import torch.nn.functional as F

class PolicyModel(nn.Module):
    def __init__(self, action_size: int, feature_dim: int):
        """
        args:
            action_size (int): 行動の次元数
            feature_dim (int): 特徴量の次元数
        """
        super().__init__()

        self.action_size = action_size

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 400),
            nn.ELU(),
            nn.Linear(400, 400),
            nn.ELU(),
        )
        self.head = nn.Linear(400, action_size)
    
    def forward(self, x):
        """
        args:
            x (torch.Tensor): 入力画像 shape=(Batch, Feature_dim)
        Returns:
            torch.Tensor: 行動の確率分布 shape=(Batch, action_size)
        """
        x = self.mlp(x)
        logits = self.head(x)
        return logits
    
    def epsilon_greedy_aciton(self, x, epsilon):
        """
        TODO: 未整備
        args:
            x (torch.Tensor): 入力画像 shape=(Batch, Feature_dim)
            epsilon (float): ε-greedyのε
        """
        probs = self.forward(x)
        if torch.rand(1) < epsilon:
            action = torch.randint(0, self.action_space, (1,))
        else:
            action = torch.distributions.Categorical(probs=probs).sample()
        return action
    
    def sample(self, x):
        """
        TODO: 未整備
        args:
            x (torch.Tensor): 入力画像 shape=(Batch, Feature_dim)
        """
        probs = self.forward(x)
        dist = torch.distributions.Categorical(probs=probs)
        return dist.sample()

class ValueModel(nn.Module):
    def __init__(self, action_size: int, feature_dim: int):
        """
        args:
            action_size (int): 行動の次元数
            feature_dim (int): 特徴量の次元数
        """
        super().__init__()

        self.action_size = action_size

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 400),
            nn.ELU(),
            nn.Linear(400, 400),
            nn.ELU(),
        )
        self.head = nn.Linear(400, 1)

    def forward(self, x):
        """
        args:
            x (torch.Tensor): 入力画像 shape=(Batch, Feature_dim)
        """
        x = self.mlp(x)
        value = self.head(x)
        return value