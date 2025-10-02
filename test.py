import gymnasium as gym
print([env.id for env in gym.envs.registry if "LunarLander" in env.id])