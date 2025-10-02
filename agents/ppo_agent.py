# Author: Sophie Zhao


from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, List

# ─────────────────────────────────────────────────────────────────────────────
# Network Definitions
# ─────────────────────────────────────────────────────────────────────────────
class MAPPOActor(nn.Module):
    """Stateless feed-forward policy network."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        logits = self.net(obs)
        return Categorical(logits=logits)

class MAPPOCritic(nn.Module):
    """Centralized value function for concatenated joint observations."""
    def __init__(self, joint_obs_dim: int, hidden: int = 256):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, joint_obs: torch.Tensor) -> torch.Tensor:
        return self.v(joint_obs)

# ─────────────────────────────────────────────────────────────────────────────
# MAPPO Agent Wrapper
# ─────────────────────────────────────────────────────────────────────────────
class MAPPOAgent:
    """
    Multi-agent PPO agent for Overcooked.
    Holds two actors (optionally weight-shared) and a central critic.
    Provides helper methods for acting and PPO updates.
    """
    def __init__(
        self,
        obs_dim: int = 96,
        act_dim: int = 6,
        hidden: int = 128,
        critic_hidden: int = 256,
        lr: float = 2.5e-4,
        shared_actor: bool = True,
        ent_coef: float = 0.01,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.ent_coef = ent_coef

        # Actors (can share weights)
        self.actor1 = MAPPOActor(obs_dim, act_dim, hidden).to(self.device)
        if shared_actor:
            self.actor2 = self.actor1  # Share weights
        else:
            self.actor2 = MAPPOActor(obs_dim, act_dim, hidden).to(self.device)

        # Centralized critic
        self.critic = MAPPOCritic(obs_dim * 2, critic_hidden).to(self.device)

        # Joint optimizer
        self.optim = torch.optim.Adam(
            list(self.actor1.parameters())
            + ([] if shared_actor else list(self.actor2.parameters()))
            + list(self.critic.parameters()),
            lr=lr,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Action selection for both agents
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def act(
        self, obs_n: List[torch.Tensor], deterministic: bool = False
    ) -> Tuple[List[int], List[float]]:
        obs1 = obs_n[0].to(self.device).unsqueeze(0)
        obs2 = obs_n[1].to(self.device).unsqueeze(0)
        dist1 = self.actor1(obs1)
        dist2 = self.actor2(obs2)
        if deterministic:
            a1 = torch.argmax(dist1.probs, dim=-1)
            a2 = torch.argmax(dist2.probs, dim=-1)
        else:
            a1 = dist1.sample()
            a2 = dist2.sample()
        logp1 = dist1.log_prob(a1)
        logp2 = dist2.log_prob(a2)
        return [a1.item(), a2.item()], [logp1.item(), logp2.item()]

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation for PPO updates
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate(
        self,
        obs_batch_n: Tuple[torch.Tensor, torch.Tensor],
        actions_batch_n: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs1, obs2 = obs_batch_n
        act1, act2 = actions_batch_n
        obs1 = obs1.to(self.device)
        obs2 = obs2.to(self.device)
        act1 = act1.to(self.device)
        act2 = act2.to(self.device)
        dist1 = self.actor1(obs1)
        dist2 = self.actor2(obs2)
        logp1 = dist1.log_prob(act1)
        logp2 = dist2.log_prob(act2)
        entropy1 = dist1.entropy()
        entropy2 = dist2.entropy()
        # Centralized critic: concatenate obs
        joint_obs = torch.cat([obs1, obs2], dim=-1)
        values = self.critic(joint_obs).squeeze(-1)  # (B,)
        logprobs = torch.stack([logp1, logp2], dim=-1)   # (B, 2)
        entropies = torch.stack([entropy1, entropy2], dim=-1)
        return logprobs, entropies, values

    # ─────────────────────────────────────────────────────────────────────────
    # PPO Update Step
    # ─────────────────────────────────────────────────────────────────────────
    def update(
        self,
        batch_dict: dict,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = None,   # Override via argument or use self.ent_coef
    ):
        """
        PPO clipped policy/value update.
        """
        ent_coef = ent_coef if ent_coef is not None else self.ent_coef

        obs1 = batch_dict['obs1'].to(self.device)
        obs2 = batch_dict['obs2'].to(self.device)
        a1   = batch_dict['actions1'].to(self.device)
        a2   = batch_dict['actions2'].to(self.device)
        old_logp = torch.stack([batch_dict['old_logp1'], batch_dict['old_logp2']], dim=-1).to(self.device)
        advantages = batch_dict['advantages'].to(self.device)
        returns    = batch_dict['returns'].to(self.device)

        # Forward pass
        logp, entropy, values = self.evaluate((obs1, obs2), (a1, a2))
        ratios = torch.exp(logp - old_logp)
        ratios_clipped = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps)

        # Policy loss (mean over both agents and batch)
        adv_expanded = advantages.unsqueeze(-1)  # (B, 1) for broadcast
        policy_loss = -torch.mean(torch.min(ratios * adv_expanded, ratios_clipped * adv_expanded))

        # Value loss (mean squared error)
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus (for exploration)
        entropy_bonus = entropy.mean()

        total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_bonus

        self.optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor1.parameters(), 0.5)
        if self.actor2 is not self.actor1:
            torch.nn.utils.clip_grad_norm_(self.actor2.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optim.step()

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy_bonus.item(),
        }