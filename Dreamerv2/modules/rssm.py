import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentStateSpaceModel(nn.Module):
    def __init__(self, latent_dim: int, n_atoms: int, hidden_dim: int, action_size: int):
        """
        args:
            latent_dim (int): 潜在変数の次元数
            n_atoms (int): 分散表現の次元数
            hidden_dim (int): 隠れ層の次元数
            action_size (int): 行動の次元数
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.n_atoms = n_atoms
        self.hidden_dim = hidden_dim

        self.linear_z_prior1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_z_prior2 = nn.Linear(hidden_dim, latent_dim*n_atoms)
        self.linear_z_post1 = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.linear_z_post2 = nn.Linear(hidden_dim, latent_dim*n_atoms)
        self.linear_h1 = nn.Linear(latent_dim*n_atoms + action_size, hidden_dim)

        self.gru_cell = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.init_h = nn.Parameter(torch.zeros(1, hidden_dim))
    
    def forward(self):
        """
        Redundunt method
        """
        raise ValueError("This method is not used")
    
    def get_initial_hidden_state(self, batch_size):
        # return torch.zeros(batch_size, self.hidden_dim)
        return self.init_h.expand(batch_size, -1)

    def sample_z_prior(self, hidden_state):
        """
        paperでいうところのTransition Predictor部分
        Args:
            hidden_state (torch.Tensor): 隠れ状態 shape=(Batch, Hidden_dim)
        Returns:
            torch.Tensor: 状態 shape=(Batch, Latent_dim, N_atoms)
            torch.Tensor: 状態の確率 shape=(Batch, Latent_dim, N_atoms)
        """
        x = F.elu(self.linear_z_prior1(hidden_state)) # 活性化関数所説？
        z_logits = F.relu(self.linear_z_prior2(x))
        z_logits = z_logits.view(-1, self.latent_dim ,self.n_atoms)
        z_probs = F.softmax(z_logits, dim=2)

        dist = torch.distributions.OneHotCategoricalStraightThrough(probs=z_probs)
        z = dist.sample()

        # Reparameterization trick
        z = z + z_probs - z_probs.detach()

        return z, z_probs

    def sample_z_post(self, hidden_state, embed_feature):
        """
        paperでいうところのRepresentation model部分
        Args:
            hidden_state (torch.Tensor): 隠れ状態 shape=(Batch, Hidden_dim)
            embed_feature (torch.Tensor): エンコードされた画像 shape=(Batch, Latent_dim)
        Returns:
            torch.Tensor: 状態 shape=(Batch, Latent_dim, N_atoms)
            torch.Tensor: 状態の確率 shape=(Batch, Latent_dim, N_atoms)
        """
        x = torch.cat([hidden_state, embed_feature], dim=1)
        x = F.elu(self.linear_z_post1(x))
        z_logits = F.relu(self.linear_z_post2(x))
        z_logits = z_logits.view(-1, self.latent_dim, self.n_atoms)
        z_probs = F.softmax(z_logits, dim=2)
        dist = torch.distributions.OneHotCategoricalStraightThrough(probs=z_probs)
        z = dist.sample()

        # Reparameterization trick
        z = z + z_probs - z_probs.detach()

        return z, z_probs

    def step_h(self, z, hidden_state, action):
        """
        paperでいうところのReccurent model部分
        h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        args:
            z (torch.Tensor): 状態 shape=(Batch, Latent_dim, N_atoms)
            hidden_state (torch.Tensor): 隠れ状態 shape=(Batch, Hidden_dim)
            action (torch.Tensor): 行動 shape=(Batch, Action_size)
        """
        z = z.view(-1, self.latent_dim*self.n_atoms)
        x = torch.cat([z, action], dim=1)
        x = F.elu(self.linear_h1(x))
        next_hidden_state = self.gru_cell(x, hidden_state)

        return next_hidden_state 