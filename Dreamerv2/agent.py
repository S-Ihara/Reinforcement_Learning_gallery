import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as D
from modules.worldmodel import WorldModel
from modules.actorcritic import PolicyModel, ValueModel

class DreamerAgent:
    def __init__(self, latent_dim: int, n_atoms: int, hidden_dim: int, action_size: int, img_channels=1):
        """
        args:
            latent_dim (int): 潜在変数の次元数
            n_atoms (int): 分散表現の次元数
            hidden_dim (int): 隠れ層の次元数
            action_size (int): 行動の次元数
            img_channels (int): 画像のチャンネル数
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.world_model = WorldModel(latent_dim=latent_dim, n_atoms=n_atoms, hidden_dim=hidden_dim, action_size=action_size, img_channels=img_channels)
        self.critic = ValueModel(action_size=action_size, feature_dim=hidden_dim + latent_dim*n_atoms)
        self.actor = PolicyModel(action_size=action_size, feature_dim=hidden_dim + latent_dim*n_atoms)

        self.world_model.to(self.device)
        self.critic.to(self.device)
        self.actor.to(self.device)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=1e-4)

        self.gamma_discount = 0.99
        self.kl_alpha = 0.5 # kl balancing
        self.kl_scale = 0.1 # kl regularizer
        self.actor_critic_init_size = 16
        self.horizon = 8 # ドリーム内でのロールアウトの長さ
        self.actor_critic_batch_size = 16 # actor critic訓練のバッチサイズ
        self.entropy_scale = 0.01 # entropy係数

        _,self.hidden_state = self.world_model.get_initial_state(1)

    def reset(self):
        """
        推論用の隠れ状態を初期化
        """
        _,self.hidden_state = self.world_model.get_initial_state(1)

    @torch.no_grad()
    def get_action(self, state, **kwargs):
        """
        状態を引数として行動を返す
        args:
            state (np.ndarray): 状態 shape=(Channel, Height, Width)
            kwargs:
                return_reconstruction (bool): 再構成画像を返すかどうか
        returns:
            int | dict[str,Any]: 行動などを返す
        """
        if state.dtype == np.float32:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        elif state.dtype == np.uint8:
            state = state / 255.0
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        #state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        embed = self.world_model.encoder(state)
        z,_ = self.world_model.rssm.sample_z_post(self.hidden_state, embed)
        z = z.view(z.size(0), -1)
        feature = torch.cat([self.hidden_state, z], dim=1)
        action = self.actor(feature)
        action = torch.argmax(action).item()

        if kwargs.get("return_reconstruction", False):
            reconstruction = self.world_model.decoder(feature)
            return {
                "action": action,
                "reconstruction": reconstruction,
            }

        return action 

    def train_worldmodel_minibatch(self, states, actions, rewards, terminals):
        """
        args:
            states (torch.Tensor): 状態 shape=(Length, Batch, Channel, Height, Width)
            actions (torch.Tensor): 行動 shape=(Length, Batch, Action_size)
            rewards (torch.Tensor): 報酬 shape=(Length, Batch, 1)
            terminals (torch.Tensor): 終了フラグ shape=(Length, Batch, 1)
        returns:
            dict[str, Any]: ロスとか
        """
        length = states.size(0)
        batch_size = states.size(1)
        
        loss = 0
        prev_z, prev_h = self.world_model.get_initial_state(batch_size)
        sampled_z = []
        sampled_h = []
        for l in range(length):
            hidden_state, z_prior, z_prior_probs, z_post, z_post_probs, img_reconstruction, reward_mean, discount_logit \
                = self.world_model(states[l], prev_z, prev_h, actions[l])
            
            sampled_z.append(z_post)
            sampled_h.append(hidden_state)

            # reconstruction loss
            img_reconstruction_loss = F.mse_loss(img_reconstruction, states[l]) # 一旦MSEで

            # reward loss
            #reward_loss = F.mse_loss(reward_mean, rewards[l])
            reward_dist = D.Normal(loc=reward_mean, scale=1.)
            reward_loss = -reward_dist.log_prob(rewards[l]).mean()

            # discount loss
            #discounts = (1. - terminals[l].unsqueeze(-1)) * self.gamma_discount
            #discount_loss = F.binary_cross_entropy_with_logits(discount_logit, discounts)
            #discount_dist = D.Normal(loc=discount_logit, scale=1.)
            discount_dist = D.Bernoulli(logits=discount_logit)
            discount_loss = -discount_dist.log_prob(terminals[l]).mean()

            # kl loss
            kl_loss = self._kl_loss(z_prior_probs, z_post_probs)

            total_loss = img_reconstruction_loss + reward_loss + discount_loss + kl_loss*self.kl_scale 
            loss = loss + (total_loss / length)
        
        self.world_model_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # grad_norm_model = torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.grad_clip_norm) # sheeprlかなんかに実装あったはず
        self.world_model_optimizer.step()

        # actor critic学習用のランダム初期点
        random_idx = np.random.choice(range(length*batch_size), self.actor_critic_init_size)
        sampled_z = torch.stack(sampled_z, dim=0).view(length*batch_size, -1)[random_idx]
        sampled_h = torch.stack(sampled_h, dim=0).view(length*batch_size, -1)[random_idx]

        return {
            "loss": loss.item(),
            "img_reconstruction_loss": img_reconstruction_loss.item(),
            "reward_loss": reward_loss.item(),
            "discount_loss": discount_loss.item(),
            "kl_loss": kl_loss.item(),
            "sampled_z": sampled_z,
            "sampled_h": sampled_h,
        }
    
    def _kl_loss(self, z_prior_probs, z_post_probs):
        """
        kl balancing付きのkl loss
        """
        kl_div1 = D.kl_divergence(
            D.OneHotCategoricalStraightThrough(probs=z_post_probs),
            D.OneHotCategoricalStraightThrough(probs=z_prior_probs.detach()),
        )

        kl_div2 = D.kl_divergence(
            D.OneHotCategoricalStraightThrough(probs=z_post_probs.detach()),
            D.OneHotCategoricalStraightThrough(probs=z_prior_probs),
        )

        kl_loss = self.kl_alpha * kl_div1.mean() + (1. - self.kl_alpha) * kl_div2.mean()
        return kl_loss
    
    def train_actor_critic(self, sampled_z, sampled_h):
        """
        actor criticの学習
        args:
            sampled_z (torch.Tensor): サンプリングされた状態 shape=(Actor_critic_init_size, Latent_dim, N_atoms)
            sampled_h (torch.Tensor): サンプリングされた隠れ状態 shape=(Actor_critic_init_size, Hidden_dim)
        returns:
            dict[str, Any]: ロスとか
        """
        imagined_trajectory = self.rollout_in_dream(sampled_z, sampled_h) # shape = (Horizon, Actor_critic_init_size, *shape)

        states = imagined_trajectory["states"]
        actions = imagined_trajectory["actions"]
        rewards = imagined_trajectory["rewards"]
        discounts = imagined_trajectory["discounts"]
        next_states = imagined_trajectory["next_states"]

        states = states.view(-1, states.size(-1))
        actions = actions.view(-1, actions.size(-1))
        rewards = rewards.view(-1, rewards.size(-1))
        next_states = next_states.view(-1, next_states.size(-1))
        discounts = discounts.view(-1, discounts.size(-1))

        targets, weights = self._compute_target(states, rewards, next_states, discounts)

        old_action_probs = self.actor(states) # shape=(Actor_critic_init_size*horizon, Action_size)
        old_action_probs = F.softmax(old_action_probs, dim=-1)
        old_action_logprobs = torch.log(old_action_probs)
        
        for _ in range(10):
            indices = torch.randint(low=0, high=states.size(0), size=(self.actor_critic_batch_size,))  # ランダムにバッチを選択
            # _states = states[indices]
            # _targets = targets[indices]
            # _actions = actions[indices]
            # _old_logprobs = old_action_logprobs[indices]
            # _weights = weights[indices]

            _states = states[indices].clone().detach().requires_grad_(True)
            _targets = targets[indices].clone().detach().requires_grad_(True)
            _actions = actions[indices].clone().detach().requires_grad_(True)
            _old_logprobs = old_action_logprobs[indices].clone().detach().requires_grad_(True)
            _weights = weights[indices].clone().detach().requires_grad_(True)

            # update value network
            v_pred = self.critic(_states)
            advantages = _targets - v_pred
            value_loss = 0.5 * advantages**2
            discount_value_loss = torch.mean(value_loss * _weights)

            self.critic_optimizer.zero_grad()
            discount_value_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            # update policy network
            action_probs = self.actor(_states)
            action_probs = F.softmax(action_probs, dim=-1) + 1e-6
            action_logprobs = torch.sum(_actions * torch.log(action_probs), dim=-1, keepdim=True)

            objective = action_logprobs * advantages

            distribution = D.OneHotCategoricalStraightThrough(probs=action_probs)
            entropy = distribution.entropy()

            policy_loss = -(objective + self.entropy_scale * entropy)
            policy_loss = torch.mean(policy_loss * _weights)

            self.actor_optimizer.zero_grad()
            #policy_loss.backward()
            self.actor_optimizer.step()

        return {
            "value_loss": discount_value_loss.item(),
            "policy_loss": policy_loss.item(),
        }
        
    
    def _compute_target(self, states, rewards, next_states, discounts):
        """
        args:
            states (torch.Tensor): 状態 shape=(Batch, Feature_dim)
            rewards (torch.Tensor): 報酬 shape=(Batch, 1)
            next_states (torch.Tensor): 次の状態 shape=(Batch, Feature_dim)
            discounts (torch.Tensor): 割引率 shape=(Batch, 1)
        """
        B,_ = states.size()
        value_next = self.critic(next_states)
        _weights = torch.cat([torch.ones_like(discounts[:1]), discounts[:-1]], dim=0)
        weights = torch.cumprod(_weights, dim=0)

        targets = torch.zeros_like(value_next)
        last_value = value_next[-1]

        for i in reversed(range(B)):
            last_value = rewards[i] + discounts[i] * last_value
            targets[i] = last_value
        
        return targets, weights


    def rollout_in_dream(self, z_init, h_init, video=False):
        """
        ドリーム内でのロールアウト
        args:
            z_init (torch.Tensor): 初期状態 shape=(Batch, Latent_dim, N_atoms)
            h_init (torch.Tensor): 初期隠れ状態 shape=(Batch, Hidden_dim)
            video (bool): ビデオ保存するかどうか
        returns:
            dict[str, torch.Tensor]: ロールアウトした結果 shape=(Horizon, Batch, *Shape)
        """
        feature = torch.cat([h_init, z_init.view(z_init.size(0), -1)], dim=1)
        states = []
        actions = []
        next_states = []

        z, h = z_init, h_init

        for t in range(self.horizon):
            action = self.actor(feature)
            action = F.softmax(action, dim=-1)
            
            states.append(feature)
            actions.append(action)

            hidden_state = self.world_model.rssm.step_h(z, h, action)
            z, _ = self.world_model.rssm.sample_z_prior(hidden_state)
            z = z.view(z.size(0), -1)

            feature = torch.cat([hidden_state, z], dim=1)
            next_states.append(feature)
        
        states = torch.stack(states, dim=0)
        actions = torch.stack(actions, dim=0)
        next_states = torch.stack(next_states, dim=0)
        rewards = self.world_model.reward_head(next_states)
        discounts = self.world_model.discount_head(next_states)
        discounts = D.Bernoulli(logits=discounts).mean

        return {
            "states": states,
            "actions": actions,
            "next_states": next_states,
            "rewards": rewards,
            "discounts": discounts,
        }
    
    def save(self, path):
        """
        モデルを保存
        args:
            path (str): 保存先のパス
        """
        torch.save({
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)
    
    def load(self, path):
        """
        モデルをロード
        args:
            path (str): ロードするパス
        """
        checkpoint = torch.load(path)
        self.world_model.load_state_dict(checkpoint["world_model"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])