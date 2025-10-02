

from __future__ import annotations

import argparse

import os

import time

from pathlib import Path
from collections import deque

import torch
import numpy as np
import matplotlib.pyplot as plt
from env.overcooked_setup import make_overcooked_env
from agents.ppo_agent import MAPPOAgent
from training.rollout import collect_rollout


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]  # 0 if terminal
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def train_one_layout(
        layout: str,
        total_steps: int = 4_000_000,
        rollout_horizon: int = 400,
        update_epochs: int = 4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        device: str = "cpu",
        plot: bool = True,
):
    reward_shaping = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5
    }

    env = make_overcooked_env(layout, reward_shaping=reward_shaping)
    agent = MAPPOAgent(device=device)

    steps_collected = 0
    episodes = 0

    # Store metrics for plotting
    soup_counts = []
    dish_pickups = []
    dropped_dishes = []
    shaped0s, shaped1s = [], []

    episode_returns = deque(maxlen=100)
    start_train = time.time()

    while steps_collected < total_steps:
        buffer, episode_return, soups, srew0, srew1 = collect_rollout(
            env, agent, rollout_horizon, deterministic=False, device=device
        )
        data = buffer.as_tensors()
        T = data["reward"].shape[0]
        steps_collected += T

        # Advantage & return
        with torch.no_grad():
            joint_obs = torch.cat([data["obs1"], data["obs2"]], dim=-1)
            values = agent.critic(joint_obs).squeeze(-1).cpu().numpy()
        adv, ret = compute_gae(
            data["reward"].cpu().numpy(),
            values,
            data["done"].cpu().numpy(),
            gamma=gamma,
            lam=lam,
        )
        data["advantages"] = torch.tensor(adv, dtype=torch.float32, device=device)
        data["returns"] = torch.tensor(ret, dtype=torch.float32, device=device)
        data["old_logp1"] = data["logp1"].clone().detach()
        data["old_logp2"] = data["logp2"].clone().detach()

        for _ in range(update_epochs):
            # agent.update(data, clip_eps=clip_eps, ent_coef=0.05)
            agent.update(data, clip_eps=clip_eps)

        episodes += 1
        episode_returns.append(episode_return)
        soup_counts.append(soups)
        shaped0s.append(srew0)
        shaped1s.append(srew1)

        # Collect info metrics (may want to sum over episode)
        ep_info_list = data["info"]  # list of dicts, one per step
        # Sum up dish pickups and dropped dishes over this episode
        # print("------------------------")
        # print(ep_stats)
        # print("------------------------")
        pickups, drops = 0, 0

        for info in ep_info_list:
            if "episode" in info:
                # This only exists at the end of the episode
                ep_stats = info["episode"]["ep_game_stats"]
                # print("------------------------")
                # print(ep_stats)
                # print("------------------------")
                pickups = sum(len(lst) for lst in ep_stats.get("dish_pickup", []))
                drops = sum(len(lst) for lst in ep_stats.get("dish_drop", []))
                break  # only need the final one
        dish_pickups.append(pickups)
        dropped_dishes.append(drops)

        if episodes % 10 == 0:
            mean_soup = np.mean(soup_counts[-10:])
            print(
                f"Ep {episodes:4d} | Return: {episode_return:.1f} | Mean Soup(10): {mean_soup:.1f} | "
                f"Soups: {soups} | Shaped0: {srew0:.1f} | Shaped1: {srew1:.1f} | "
                f"DishPickups: {pickups} | Drops: {drops}"
            )

    # Save model and raw metrics
    save_dir = Path("results") / "metrics"
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / f"soup_counts_{layout}.npy", soup_counts)
    np.save(save_dir / f"dish_pickups_{layout}.npy", dish_pickups)
    np.save(save_dir / f"dropped_dishes_{layout}.npy", dropped_dishes)


    if plot:
        N = 20  # moving average window
        x = np.arange(len(soup_counts))

        def moving_avg(y): return np.convolve(y, np.ones(N) / N, mode="valid")

        plt.figure(figsize=(12, 6))
        plt.plot(x[N - 1:], moving_avg(soup_counts), label="Soups Delivered (Moving Avg)", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Soups Delivered")
        plt.title(f"Soups Delivered per Episode: {layout}")
        plt.legend()
        plt.savefig(save_dir / f"soups_{layout}.png")
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(x[N - 1:], moving_avg(dish_pickups), label="Dish Pickups", color='g')
        plt.plot(x[N - 1:], moving_avg(dropped_dishes), label="Dropped Dishes", color='r')
        plt.xlabel("Episode")
        plt.ylabel("Count per Episode")
        plt.title(f"Dish Pickups & Drops: {layout}")
        plt.legend()
        plt.savefig(save_dir / f"pickups_drops_{layout}.png")
        plt.show()

    # Save model
    save_path = Path("results/models") / f"mappo_{layout}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.actor1.state_dict(), save_path)
    print(f"Finished training {layout} — model saved to {save_path}.")
    print(f"Saved metrics to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #layout = "cramped_room"
    # layout = "coordination_ring"
    layout = "counter_circuit_O_1order"

    start = time.time()

    train_one_layout(layout=layout)

    print(f"Training completed in {(time.time() - start) / 60:.1f} min.")
