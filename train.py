import os
import random

import gymnasium as gym
import numpy as np
import torch

from ddpg import DDPGAgent
from memory import ReplayBuffer
from gymnasium.spaces import Box


def set_seed(env, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)

def train(
    env_name: str = "LunarLanderContinuous-v3",
    episodes: int = 500,
    max_steps: int = 1_000,
    start_steps: int = 10_000,
    batch_size: int = 64,
    buffer_capacity: int = 100_000,
    eval_every: int = 25,
    save_path: str | None = "models/ddpg_lander.pt",
):
    # Environment
    env = gym.make(env_name)
    start_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_space: Box = env.action_space
    max_action = float(env.action_space.high[0])

    # Agent and buffer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DDPGAgent(start_dim, action_dim, max_action=max_action, device=device)
    buffer = ReplayBuffer(buffer_capacity, device=device)

    # Seeding
    set_seed(env)

    # Training Loop
    total_steps = 0
    episode_rewards: list[float] = []
    eval_history = []

    for ep in range(1, episodes+1):
        state, _ = env.reset()
        ep_reward = 0.0

        for t in range(max_steps):
            if total_steps < start_steps:
                # random exploration to fill the buffer
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, explore=True)

            next_state, reward, done, truncated, _ = env.step(action)
            terminal = done or truncated

            buffer.add(state, action, reward, next_state, terminal)
            state = next_state
            ep_reward += reward

            total_steps+=1

            # update agent after collection enough samples
            if len(buffer) >= batch_size and total_steps > start_steps:
                batch = buffer.sample(batch_size)
                critic_loss, actor_loss = agent.update(batch)

            if terminal:
                break
        episode_rewards.append(ep_reward)
        print(f"Episode {ep:4d} | Reward: {ep_reward:8.2f} | Buffer: {len(buffer):6d}")

        # periodic evaluation (no exploration noise)
        if ep % eval_every == 0:
            eval_reward = evaluate(agent, env)
            print(f"--> Eval after {ep} episodes: {eval_reward:.2f}")
            eval_history.append(eval_reward)

        # optional save
        if save_path and ep%eval_every == 0:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            agent.save(save_path)

    env.close()

    # save array for plotting
    os.makedirs("logs", exist_ok=True)
    np.save("logs/train_rewards.npy", np.array(episode_rewards))

    # Keep exactly the last 100 evaluation runs (repeat eval if you need more)
    if eval_history:
        np.save("logs/eval_rewards.npy",
                np.array(eval_history[-100:]))

def evaluate(agent: DDPGAgent, env: gym.Env, runs: int=5, max_steps:int = 1_000) -> float:
    total = 0.0
    for _ in range(runs):
        state, _ = env.reset()
        ep_reward = 0.0
        for _ in range(max_steps):
            action = agent.select_action(state, explore=False)
            state, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            if done or truncated:
                break
        total += ep_reward
    return total/runs


if __name__ == "__main__":
    train()


