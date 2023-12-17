import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    VAEencoder
    入力画像解像度は64x64で固定（可変にするとdeconv周りが大変なのでとりあえず）
    """
    def __init__(self, img_channels, latent_size):
        """
        Args:
            img_channels (int): 入力画像のチャンネル数
            latent_size (int): 潜在変数の次元数
        """
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(2*2*256, latent_size)
        self.fc_logsigma = nn.Linear(2*2*256, latent_size)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma
    
class Decoder(nn.Module):
    """
    VAEdecoder 出力解像度は64x64
    """
    def __init__(self, img_channels, latent_size):
        """
        Args:
            img_channels (int): 出力画像のチャンネル数
            latent_size (int): 潜在変数の次元数
        """
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(latent_size, 1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 6, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        reconstruction = self.decoder(x)
        return reconstruction

class VAE(nn.Module):
    """Variational Autoencoder
    """
    def __init__(self, img_channels, latent_size):
        """
        Args:
            img_channels (int): 入出力画像のチャンネル数
            latent_size (int): 潜在変数の次元数
        """
        super(VAE, self).__init__()
        self.encoder = Encoder(img_channels, latent_size)
        self.decoder = Decoder(img_channels, latent_size)

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.decoder(z)
        return recon_x, mu, logsigma