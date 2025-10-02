import itertools
import os
import gymnasium as gym
import numpy as np
import torch
from typing import Dict, List

from ddpg import DDPGAgent
from memory import ReplayBuffer


def run_experiment(
    actor_lr: float,
    critic_lr: float,
    tau: float,
    batch_size: int,
    episodes: int = 400,
    eval_runs: int = 10,
    seed: int = 42,
) -> float:
    """Train a DDPG agent with given hyper-params and return the mean eval reward."""
    env = gym.make("LunarLanderContinuous-v3")
    # reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)

    # dims and limits
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DDPGAgent(
        state_dim,
        action_dim,
        max_action=max_action,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        tau=tau,
        device=device,
    )
    buffer = ReplayBuffer(100_000, device=device)

    # training hyperparams
    total_steps = 0
    start_steps = 10_000
    max_steps = 1_000

    # training loop
    for ep in range(episodes):
        state, _ = env.reset()
        for _ in range(max_steps):
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, explore=True)
            next_state, reward, done, truncated, _ = env.step(action)
            buffer.add(state, action, reward, next_state, done or truncated)
            state = next_state
            total_steps += 1
            if len(buffer) >= batch_size and total_steps > start_steps:
                batch = buffer.sample(batch_size)
                agent.update(batch)
            if done or truncated:
                break
    # evaluation: multiple runs for stable average
    eval_returns = []
    for _ in range(eval_runs):
        state, _ = env.reset()
        ep_ret = 0.0
        for _ in range(max_steps):
            action = agent.select_action(state, explore=False)
            state, reward, done, truncated, _ = env.step(action)
            ep_ret += reward
            if done or truncated:
                break
        eval_returns.append(ep_ret)
    env.close()
    return float(np.mean(eval_returns))


if __name__ == "__main__":
    # Hyper-param grids
    # actor_lrs   = [1e-5, 1e-4, 5e-4]
    # critic_lrs  = [1e-4, 1e-3, 5e-3]
    actor_lrs = [5e-4]
    critic_lrs = [1e-3]
    # taus = [5e-4, 1e-3, 5e-3]
    taus = [1e-3, 5e-3, 1e-2]
    # batch_sizes = [32, 64, 128]
    batch_sizes = [64]

    # Collect results
    results: Dict[str, List] = {"actor_lr": [], "critic_lr": [], "tau": [], "batch_size": [], "metric": []}

    for a_lr, c_lr, tau_val, bs in itertools.product(actor_lrs, critic_lrs, taus, batch_sizes):
        avg_reward = run_experiment(a_lr, c_lr, tau_val, bs)
        results["actor_lr"].append(a_lr)
        results["critic_lr"].append(c_lr)
        results["tau"].append(tau_val)
        results["batch_size"].append(bs)
        results["metric"].append(avg_reward)
        print(f"a_lr={a_lr:.1e}, c_lr={c_lr:.1e}, tau={tau_val:.1e}, bs={bs} -> {avg_reward:.2f}")

    # Save for later visualization
    os.makedirs("logs", exist_ok=True)
    np.save("logs/hparam_sweep.npy", results)

    # Plotting utilities
    import matplotlib.pyplot as plt


    def plot_effect(param: str, xlabel: str, log_scale: bool = False):
        vals = sorted(set(results[param]))
        means = [np.mean([m for p, m in zip(results[param], results["metric"]) if p == v]) for v in vals]
        plt.figure(figsize=(6, 4))
        plt.plot(vals, means, marker='o', linestyle='-')
        if log_scale:
            plt.xscale('log')
        # --- show every tested value explicitly --- #
        tick_labels = [f"{v:.0e}" if log_scale else str(v) for v in vals]
        plt.xticks(vals, tick_labels)
        plt.xlabel(xlabel)
        plt.ylabel('Avg Eval Reward')
        plt.title(f'{param} sweep')
        plt.grid(alpha=0.3, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f'results/hparam_{param}.png', dpi=150, bbox_inches='tight')
        plt.show()

    # Generate plots with all tick values
    # plot_effect('actor_lr',   'actor_lr',   log_scale=True)
    # plot_effect('critic_lr',  'critic_lr',  log_scale=True)
    plot_effect('tau',        'tau',        log_scale=True)
    # plot_effect('batch_size', 'batch_size', log_scale=False)

    print("Hyperparameter sweep complete; plots saved in results/")
