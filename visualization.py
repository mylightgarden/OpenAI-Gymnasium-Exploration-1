import os
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np


def load_rewards(
        train_path: str = "logs/train_rewards.npy",
        eval_path: str | None = "logs/eval_rewards.npy",
) -> Dict[str, np.ndarray]:
    """Return a dict containing reward arrays if the files exist."""
    out: Dict[str, np.ndarray] = {}
    if os.path.exists(train_path):
        out["train"] = np.load(train_path)
    if eval_path and os.path.exists(eval_path):
        out["eval"] = np.load(eval_path)
    return out


# 1· Reward per training episode (learning curve)
def plot_training_curve(train_rewards: np.ndarray, save_to: str | None = None):
    plt.figure(figsize=(8, 4))
    plt.plot(train_rewards, label="Episode Reward", alpha=0.5)
    # Optional smoothing for readability
    if len(train_rewards) > 20:
        smoothed = np.convolve(train_rewards, np.ones(20) / 20, mode="valid")
        plt.plot(range(19, len(train_rewards)), smoothed, label="Moving Avg (20)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training reward per episode")
    plt.grid(alpha=0.3)
    plt.legend()
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.show()


# 2· Reward for 100 evaluation episodes after training
def plot_eval_episodes(eval_rewards: np.ndarray, save_to: str | None = None):
    plt.figure(figsize=(8, 4))
    plt.plot(eval_rewards, marker="o", linestyle="-")
    plt.xlabel("Episode (Evaluation)")
    plt.ylabel("Reward")
    plt.title("Trained agent: reward per episode (100 runs)")
    plt.grid(alpha=0.3)
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.show()


# 3· Hyper‑parameter sweep visualization (flexible helper)
def plot_hparam_effects(
        results: Dict[str, List[float]],
        ylabel: str = "Avg Reward",
        save_to: str | None = None,
):

    plt.figure(figsize=(8, 4))
    y = results.get("metric")
    for hparam, x_vals in results.items():
        if hparam == "metric":
            continue
        plt.plot(x_vals, y, marker="o", label=hparam)
    plt.xlabel("Hyper‑parameter value (index‑aligned)")
    plt.ylabel(ylabel)
    plt.title("Effect of hyper‑parameters on performance")
    plt.legend()
    plt.grid(alpha=0.3)
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    data = load_rewards()

    if "train" in data:
        plot_training_curve(data["train"], save_to="results/train_curve.png")

    if "eval" in data:
        plot_eval_episodes(data["eval"], save_to="results/eval_curve.png")


