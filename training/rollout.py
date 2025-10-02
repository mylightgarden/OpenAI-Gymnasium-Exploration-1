
from __future__ import annotations

from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import torch


# Buffer helper

class RolloutBuffer:

    def __init__(self, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.storage: Dict[str, List[Any]] = defaultdict(list)

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self.storage[k].append(v)

    def as_tensors(self) -> Dict[str, torch.Tensor] | Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in self.storage.items():
            if k == "info":
                out[k] = v
                continue

            sample = v[0]
            if isinstance(sample, torch.Tensor):
                out[k] = torch.stack(v).to(self.device)
            elif isinstance(sample, (np.ndarray, list)):
                out[k] = torch.tensor(v, dtype=torch.float32, device=self.device)
            elif isinstance(sample, (int, np.integer)):
                out[k] = torch.tensor(v, dtype=torch.long, device=self.device)
            elif isinstance(sample, (float, np.floating)):
                out[k] = torch.tensor(v, dtype=torch.float32, device=self.device)
            else:
                raise TypeError(
                    f"Unsupported buffer data type for key '{k}': {type(sample)}"
                )
        return out


# Roll-out collection

def _to_vec(obs, featurize_fn):

    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32, copy=False)
    if isinstance(obs, dict):
        if "obs" in obs:                       # Gym wrapper format
            return np.asarray(obs["obs"], dtype=np.float32)
        if featurize_fn is not None:
            return np.asarray(featurize_fn(obs), dtype=np.float32)
    if isinstance(obs, str):
        if featurize_fn is not None:
            return np.asarray(featurize_fn(obs), dtype=np.float32)
        else:
            print(f"Warning: Received string observation '{obs}' but no featurize_fn available")
    raise TypeError(f"Unsupported observation type: {type(obs)}")

@torch.no_grad()
def collect_rollout(
        env,
        agent,
        horizon: int = 400,
        deterministic: bool = False,
        device: str | torch.device = "cpu",
        use_shaped_reward: bool = True,
) -> tuple[RolloutBuffer, float, int, float, float]:
    buffer = RolloutBuffer(device=device)
    episode_return = 0.0
    num_soups_made = 0
    shaped_rewards_agent0 = 0.0
    shaped_rewards_agent1 = 0.0

    reset_out = env.reset()
    obs_n = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    if isinstance(obs_n, dict) and 'both_agent_obs' in obs_n:
        obs_n = obs_n['both_agent_obs']
    featurize_fn = getattr(env.unwrapped, "featurize_fn", None)

    for _ in range(horizon):
        obs_vecs = [_to_vec(o, featurize_fn) for o in obs_n]
        obs_t = [torch.as_tensor(v, dtype=torch.float32, device=device) for v in obs_vecs]
        actions_n, logps_n = agent.act(obs_t, deterministic=deterministic)
        if isinstance(actions_n, torch.Tensor):
            actions_n = actions_n.cpu().tolist()
        step_out = env.step(actions_n)
        if len(step_out) == 5:
            next_obs_n, reward, term, trunc, info = step_out
            done = term or trunc
        else:
            next_obs_n, reward, done, info = step_out
        if isinstance(next_obs_n, dict) and 'both_agent_obs' in next_obs_n:
            next_obs_n = next_obs_n['both_agent_obs']


        # --------- Align shaped rewards to the correct agent indices ---------
        r_shaped = info.get("shaped_r_by_agent", [0.0, 0.0])
        # Align shaped rewards based on env.agent_idx (see project docs!)
        if hasattr(env, "agent_idx") and env.agent_idx:
            r_shaped_0 = r_shaped[1]
            r_shaped_1 = r_shaped[0]
        else:
            r_shaped_0 = r_shaped[0]
            r_shaped_1 = r_shaped[1]
        shaped_rewards_agent0 += r_shaped_0
        shaped_rewards_agent1 += r_shaped_1

        # --------- Soup counting ---------
        if isinstance(reward, (list, tuple)):
            r_scalar = sum(reward)
        else:
            r_scalar = reward
        num_soups_made += int(r_scalar / 20)

        # --------- Choose reward to use for learning ---------
        if use_shaped_reward and "shaped_r_by_agent" in info:
            reward_to_use = r_scalar * 20 + r_shaped_0 + r_shaped_1
            # reward_to_use = r_shaped_0 + r_shaped_1
        else:
            reward_to_use = r_scalar * 20

        # Fairness: reward if both agents earning rewards
        if (r_shaped_0 > 0) and (r_shaped_1 > 0):
            fairness_bonus = 0.01  # Or any small bonus you want
        else:
            fairness_bonus = 0.0
        reward_to_use += fairness_bonus

        buffer.add(
            obs1=obs_t[0],
            obs2=obs_t[1],
            actions1=torch.tensor(actions_n[0], device=device),
            actions2=torch.tensor(actions_n[1], device=device),
            logp1=torch.tensor(logps_n[0], device=device),
            logp2=torch.tensor(logps_n[1], device=device),
            reward=torch.tensor(reward_to_use, dtype=torch.float32, device=device),
            done=torch.tensor(done, dtype=torch.float32, device=device),
            info=info
        )

        episode_return += reward_to_use
        obs_n = next_obs_n
        if done:
            break

    return buffer, episode_return, num_soups_made, shaped_rewards_agent0, shaped_rewards_agent1
