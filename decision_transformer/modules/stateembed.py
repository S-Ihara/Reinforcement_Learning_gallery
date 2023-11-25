import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d_size_out(self,size,kernel_size,stride):
    """util function to calculate output of conv2d"""
    return (size - (kernel_size - 1) - 1) // stride  + 1

class StateEmbedding(nn.Module):
    def __init__(self,observaiton_space: tuple[int,int,int], embed_dim: int, backborn: str = "miniCNN"):
        """
        Args:
            observation_space (tuple): 状態空間の次元 (channel,height,width)
            embed_dim (int): 埋め込み次元
            backborn (str): CNNのバックボーンに何を使うか
                現在はminiCNNのみ
        """
        super(StateEmbedding,self).__init__()
        assert observation_space[1] == observation_space[2], "画像は正方形にしか対応していません"
        self.observation_space = observation_space

        if backborn == "miniCNN":
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(observation_space[0],32,kernel_size=8,stride=4),
                nn.ReLU(),
                nn.Conv2d(32,64,kernel_size=4,stride=2),
                nn.ReLU(),
                nn.Conv2d(64,64,kernel_size=3,stride=1)
            )
        else:
            raise NotImplementedError
        convwh = conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(observation_space[1],8,4),4,2),3,1)
        linear_input_size = (convwh**2)*64
        self.feature_embed = nn.Sequential(
            nn.Linear(linear_input_size,embed_dim),
            nn.Tanh(),
        )

    def forward(self,x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0),-1)
        x = self.feature_embed(x)
        return x