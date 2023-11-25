import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.stateembed import StateEmbedding
from modules.transformer_modules import GPTBlocks

"""
実装上の気になったポイント
- embedで最後にはtanhを使う (-1~1にしたい？)
- positional embeddingが謎
"""

class DecisionTransformer(nn.Module):
    def __init__(self,observation_space: tuple, num_actions: int, max_timesteps: int,
                 embed_dim: int, max_context_length: int,
                 num_blocks: int, num_heads: int, **kwargs):
        """
        Args:
            observation_space (tuple): 状態空間の次元 (C,H,W) or (D,)
            num_actions (int): 行動空間の次元
            embed_dim (int): 埋め込み次元
            max_context_length (int): 最大コンテキスト長
            # 以下kwargs
            num_blocks (int): ブロック数
        """
        super(DecisionTransformer,self).__init__()

        self.observation_space = observation_space
        self.num_actions = num_actions
        self.embed_dim = embed_dim
        self.max_context_length = max_context_length

        self.return_embedding = nn.Sequential(
            nn.Linear(1,self.embed_dim),
            nn.Tanh(),
        )
        # TODO: backbornの選択
        self.state_embedding = StateEmbedding(observation_space,embed_dim)
        self.action_embedding = nn.Embedding(num_actions,embed_dim)

        self.time_embedding = nn.Embedding(max_timesteps,embed_dim)

        #self.positional_embedding = PositionalEmbedding(max_context_length,embed_dim)

        self.embed_layer_norm = nn.LayerNorm(embed_dim)

        ff_hidden_dim = embed_dim * 4
        self.blocks = nn.ModuleList([
            GPTBlocks(embed_dim=embed_dim,num_heads=num_heads,ff_hidden_dim=ff_hidden_dim,dropout=dropout)
            for _ in range(num_blocks)
        ])

        self.out_layer_norm = nn.LayerNorm(embed_dim)

        if len(observation_space) == 3:
            self.state_head = nn.Linear(embed_dim,observation_space[1]**2 * observation_space[0])
        elif len(observation_space) == 1:
            self.state_head = nn.Linear(embed_dim,observation_space[0])
        
        self.action_head = nn.Linear(embed_dim,num_actions)
        self.return_head = nn.Linear(embed_dim,1)
    
    def forward(self,returns_to_go,states,actions,timestep):
        """
        Args:
            returns_to_go (torch.Tensor): [B,L,1]
            states (torch.Tensor): [B,L,observation_space]
            actions (torch.Tensor): [B,L,1]
            timestep (torch.Tensor): [B,1,1]
        """
        batch_size, context_length,m_ = returns_to_go.size()

        rtg_embeddings = self.return_embedding(returns_to_go.squeeze()) # [B,L,embed_dim]
        state_embeddings = self.state_embedding(states) # [B,L,embed_dim]
        action_embeddings = self.action_embedding(actions.squeeze()) # [B,L,embed_dim]
        time_embeddings = self.time_embedding(timestep)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        rtg_embeddings = rtg_embeddings + time_embeddings

        # positional embeddingをもしもするならここで

        # sequenceを(R_1,s_1,a_1,R_2,s_2,a_2,...)のようにする
        sequence = torch.stack([rtg_embeddings,state_embeddings,action_embeddings],dim=1).permute(0,2,1,3).reshape(batch_size,3*context_length,self.embed_dim) # [B,3*L,embed_dim]
        sequence = self.embed_layer_norm(sequence)

        for block in self.blocks:
            sequence = block(sequence)

        sequence = self.out_layer_norm(sequence)

        rtg_predict = self.return_head(sequence[:,0::3,:]) # [B,L,1]
        state_predict = self.state_head(sequence[:,1::3,:]) # [B,L,C*H*W] or [B,L,D]
        action_predict = self.action_head(sequence[:,2::3,:]) # [B,L,num_actions]

        return rtg_predict,state_predict,action_predict
        


        



