import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentStateSpaceModel(nn.Module):
    """
    Recurrent State Space Model
    """
    def __init__(self,latent_dim: int, n_atoms: int):
        """
        Args:
            latent_dim (int): 潜在変数の次元数
            n_atoms (int): 分散表現の次元数
        """
        self.latent_dim = latent_dim
        self.n_atoms = n_atoms

        self.unit = 600

        self.linear_z_prior1 = nn.Linear(self.latent_dim, self.unit)
        self.linear_z_prior2 = nn.Linear(self.unit, self.unit)

        self.grn_cell = nn.GRUCell(self.unit, self.unit)


