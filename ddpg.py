import copy
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from actor_critic import Actor, Critic


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class GaussianNoise:
    def __init__(self, action_dim, sigma=0.1):
        self.sigma = sigma
        self.action_dim = action_dim

    def sample(self):
        return np.random.normal(0, self.sigma, size=self.action_dim)


class DDPGAgent:
    def __init__(self,
                state_dim: int,
                action_dim: int,
                max_action: float = 1.0,
                actor_lr: float = 1e-4,
                critic_lr: float = 1e-3,
                gamma: float = 0.99,
                tau: float = 0.005,  # soft update rate
                device: torch.device | str = "cpu"
                ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # exploration noise
        self.noise = GaussianNoise(action_dim)

        # Loss for critic
        self.mse = nn.MSELoss()

    # return an action in range [-max_action, max_action]
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        # .unsqueeze(0) adds a fake “batch” dimension at index 0, turning shape (state_dim,) → (1, state_dim).
        # Neural nets in PyTorch expect a batch dimension even for single examples.

        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]

        if explore:
            action += self.noise.sample()

        return np.clip(action, -self.max_action, self.max_action)

    def update(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[float, float]:
        states, actions, rewards, next_states, dones = batch

        # update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_target_next = self.critic_target(next_states, next_actions)
            q_target = rewards + self.gamma * (1.0 - dones) * q_target_next

        q_current = self.critic(states, actions)
        critic_loss = self.mse(q_current, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # software update target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return critic_loss.item(), actor_loss.item()

    # Utilities
    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict()
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target = copy.deepcopy(self.actor)
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target = copy.deepcopy(self.critic)
