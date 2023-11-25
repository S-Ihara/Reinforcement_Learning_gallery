import torch
import torch.nn as nn
import torchvision

def conv2d_size_out(self,size,kernel_size,stride):
    return (size - (kernel_size - 1) - 1) // stride  + 1

class SimpleCNN(nn.Module):
    def __init__(self,observation_space: tuple[int,int,int]):
        """
        Args:
            observation_space (tuple): 状態空間のtensor shape (channel,height,width)
        """
        super(SimpleCNN,self).__init__()
        assert observation_space[1] == observation_space[2], "画像は正方形にしか対応していません"

        convwh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(observation_space[1],8,4),4,2),3,1)
        self.output_feature_dim = convwh**2*64

        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(observation_space[0],32,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1)

    def forward(self,x):
        """
        Returns:
            torch.Tensor: 画像特徴量(B,dim)
        """
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        x = x.view(x.size(0),-1)
        return x 

    @property
    def output_feature_dim(self):
        return self.output_feature_dim

