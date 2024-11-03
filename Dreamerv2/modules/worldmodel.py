import torch 
import torch.nn as nn 

from modules.vae import Encoder, Decoder
from modules.rssm import RecurrentStateSpaceModel

class WorldModel(nn.Module):
    def __init__(self, latent_dim: int, n_atoms: int, hidden_dim: int, action_size: int, img_channels: int=1):
        """
        args:
            latent_dim (int): 潜在変数の次元数
            n_atoms (int): 分散表現の次元数
            hidden_dim (int): 隠れ層の次元数
            action_size (int): 行動の次元数
            img_channels (int): 画像のチャンネル数
        """
        super().__init__()
        
        self.laten_dim = latent_dim
        self.n_atoms = n_atoms
        self.hidden_dim = hidden_dim
        self.action_size = action_size  
        self.feature_dim = hidden_dim + latent_dim*n_atoms

        self.encoder = Encoder(latent_size=latent_dim, img_channels=img_channels)
        self.decoder = Decoder(latent_size=self.feature_dim, img_channels=img_channels)
        self.rssm = RecurrentStateSpaceModel(latent_dim=latent_dim, n_atoms=n_atoms, hidden_dim=hidden_dim, action_size=action_size)

        self.reward_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.discount_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.z_init = nn.Parameter(torch.zeros(1, latent_dim, n_atoms))
        

    def forward(self, x, prev_z, prev_h, prev_a):
        """
        args:
            x (torch.Tensor): 入力画像 shape=(Batch, Channel, Height, Width)
            prev_z (torch.Tensor): 前の状態 shape=(Batch, Latent_dim, N_atoms)
            prev_h (torch.Tensor): 前の隠れ状態 shape=(Batch, Hidden_dim)
            prev_a (torch.Tensor): 前の行動 shape=(Batch, Action_size)
        """
        embed = self.encoder(x)

        hidden_state = self.rssm.step_h(prev_z, prev_h, prev_a)

        z_prior, z_prior_probs = self.rssm.sample_z_prior(hidden_state)
        z_post, z_post_probs = self.rssm.sample_z_post(hidden_state, embed)
        # z_post = z_post.view(z_post.size(0), self.feature_dim)
        z_post = z_post.view(z_post.size(0), -1)

        feature = torch.cat([hidden_state, z_post], dim=1)

        img_reconstruction = self.decoder(feature)
        reward_mean = self.reward_head(feature)
        discount_logit = self.discount_head(feature)

        return (hidden_state, z_prior, z_prior_probs, z_post, z_post_probs, img_reconstruction, reward_mean, discount_logit)
    
    def get_initial_state(self, batch_size):
        """
        args:
            batch_size (int): バッチサイズ
        returns:
            torch.Tensor: z shape=(Batch, Latent_dim, N_atoms)
            torch.Tensor: 隠れ状態 shape=(Batch, Hidden_dim)
        """
        z_init = self.z_init.expand(batch_size, -1, -1)
        h_init = self.rssm.get_initial_hidden_state(batch_size)
        return z_init, h_init