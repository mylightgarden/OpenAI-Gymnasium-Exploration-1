# Author: Sophie Zhao

import torch
import numpy as np
import matplotlib.pyplot as plt
from env.overcooked_setup import make_overcooked_env
from agents.ppo_agent import MAPPOAgent

LAYOUTS = [
    ("cramped_room", "results/models/mappo_cramped_room.pt"),
    ("coordination_ring", "results/models/mappo_coordination_ring.pt"),
    ("counter_circuit_O_1order", "results/models/mappo_counter_circuit_O_1order.pt"),
]
DEVICE = "cpu"
EPISODES = 100

def evaluate(layout, model_path, episodes=100, device="cpu"):
    env = make_overcooked_env(layout)
    agent = MAPPOAgent(device=device)
    agent.actor1.load_state_dict(torch.load(model_path, map_location=device))
    agent.actor1.eval()
    agent.actor2.eval()

    soups_per_episode = []

    for ep in range(episodes):
        obs_n = env.reset()
        obs1, obs2 = obs_n['both_agent_obs']
        done = False
        soups = 0

        while not done:
            obs1_tensor = torch.tensor(obs1, dtype=torch.float32, device=device).unsqueeze(0)
            obs2_tensor = torch.tensor(obs2, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                dist1 = agent.actor1(obs1_tensor)
                dist2 = agent.actor2(obs2_tensor)
                # Try both sampled and greedy:
                act1 = dist1.sample().item()
                act2 = dist2.sample().item()

            actions = [act1, act2]
            obs_n, reward, done, info = env.step(actions)
            obs1, obs2 = obs_n['both_agent_obs']

            if done:
                if "episode" in info:
                    soups = info["episode"].get("ep_sparse_r", 0) // 20  # Each soup is +20
                else:
                    soups = int(reward) // 20
        soups_per_episode.append(soups)

        if done:
            print("INFO AT END OF EPISODE:", info)
            if "episode" in info:
                print("episode:", info["episode"])
            print("reward at episode end:", reward)
    return soups_per_episode

def plot_results_bar(results, layouts):
    # Calculate mean soups per episode for each layout
    means = [np.mean(soups) for soups in results]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(layouts, means, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.ylabel("Average Soups Made per Episode")
    plt.title("Average Soup-making Ability Across Layouts (Evaluation)")
    plt.ylim(0, max(means) * 1.1 if max(means) > 0 else 1)

    # Annotate values on bars
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{mean:.2f}", ha="center", va="bottom", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/soups_eval_bar.png")
    plt.show()

if __name__ == "__main__":
    all_results = []
    for layout, model_path in LAYOUTS:
        print(f"Evaluating {layout} using model {model_path} ...")
        soups = evaluate(layout, model_path, episodes=EPISODES, device=DEVICE)
        all_results.append(soups)
    plot_results_bar(all_results, [l for l, _ in LAYOUTS])