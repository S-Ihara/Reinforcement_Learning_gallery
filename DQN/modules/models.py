import torch
import torch.nn as nn
import torchvision

from .VisionTransformer import VisionTransformer

devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleQNet(torch.nn.Module):
    def __init__(self,observation_space: tuple[int],num_actions: int):
        """観測がベクトルで行動が離散値のQ関数
        Args:
            observation_space (tuple[int]): 入力の次元数
            num_actions (int): 行動の種類数
        """
        super(SimpleQNet,self).__init__()
        self.layers = torch.nn.Sequential(
              torch.nn.Linear(observation_space[0],100),
              torch.nn.Tanh(),
              torch.nn.Linear(100,100),
              torch.nn.Tanh(),
              torch.nn.Linear(100,num_actions),
        )
        ### TODO kernel initialize

    def forward(self,x):
        x = x.to(devices)
        return self.layers(x)

class CNNQNet(torch.nn.Module):
    def __init__(self,observation_space: tuple[int,int,int],num_actions: int):
        """観測が画像で行動が離散値のQ関数
        Args:
            observation_space (tuple[int,int,int]): 入力の次元数 (channel,height,width)
            num_actions (int): 行動の種類数
        """
        super(CNNQNet,self).__init__()
        assert observation_space[1] == observation_space[2], "画像は正方形にしか対応していません"
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(observation_space[0],32,kernel_size=8,stride=4),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1)
        )
        convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(observation_space[1],8,4),4,2),3,1)
        convh = convw # 正方形なので
        linear_input_size = convw*convh*64
        self.head = nn.Sequential(
            nn.Linear(linear_input_size,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,num_actions)
        )

    def forward(self,x):
        x = x.to(devices)
        x = self.feature_extractor(x)
        x = x.view(x.size(0),-1)
        x = self.head(x)
        return x

    def conv2d_size_out(self,size,kernel_size,stride):
        return (size - (kernel_size - 1) - 1) // stride  + 1
    

class ResnetQNet(torch.nn.Module):
    def __init__(self,observation_space: tuple[int,int,int],num_actions: int, train_resnet: bool = False):
        """観測が画像で行動が離散値のQ関数
        Args:
            observation_space (tuple[int,int,int]): 入力の次元数 (channel,height,width)
            num_actions (int): 行動の種類数
            train_resnet (bool): resnetの学習を行うかどうか
        """
        super(ResnetQNet,self).__init__()
        assert observation_space[1] == observation_space[2], "画像は正方形にしか対応していません"
        self.feature_extractor = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        in_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(in_features,num_actions)
        )
        self.train_resnet = train_resnet

    def forward(self,x):
        x = x.to(devices)
        # imagenet normailization
        x = x - torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(devices)
        x = x / torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(devices)
        if self.train_resnet:
            with torch.no_grad():
                x = self.feature_extractor(x)
        else:
            x = self.feature_extractor(x)
        x = x.view(x.size(0),-1)
        x = self.head(x)
        return x
    
class VitQNet(torch.nn.Module):
    def __init__(self,observation_space: tuple[int,int,int],num_actions: int):
        """観測が画像で行動が離散値のQ関数
        Args:
            observation_space (tuple[int,int,int]): 入力の次元数 (channel,height,width)
            num_actions (int): 行動の種類数
        """
        super(VitQNet,self).__init__()
        assert observation_space[1] == observation_space[2], "画像は正方形にしか対応していません"
        self.feature_extractor = VisionTransformer(
            image_size=observation_space[1],
            patch_size=8,
            in_channel=observation_space[0],
            dim=128,
            hidden_dim=128*4,
            num_heads=8,
            num_blocks=4,
            num_classes=num_actions,
        )

    def forward(self,x):
        x = x.to(devices)
        x = self.feature_extractor(x)
        return x    