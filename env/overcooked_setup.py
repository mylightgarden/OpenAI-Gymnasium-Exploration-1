# Author: Sophie Zhao

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
import gym

# Layouts available in the assignment
LAYOUTS = {
    "cramped_room": "cramped_room",
    "coordination_ring": "coordination_ring",
    "counter_circuit": "counter_circuit_o_1order"
}


def make_overcooked_env(layout_name: str, horizon: int = 400,
                        reward_shaping: dict = None, info_level: int = 0):
    """
    Initializes OvercookedEnv wrapped in gym with the specified layout and reward shaping.
    """
    mdp = OvercookedGridworld.from_layout_name(layout_name, rew_shaping_params=reward_shaping)
    print(reward_shaping)
    base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
    env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

    base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=info_level)

    # Gym wrapper
    env = gym.make("Overcooked-v0", base_env=base_env,
                   featurize_fn=base_env.featurize_state_mdp)

    return env