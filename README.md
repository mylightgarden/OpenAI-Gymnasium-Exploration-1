# OpenAI-Gymnasium-Exploration-1
Teaching an agent to land on the moon — with Deep Deterministic Policy Gradient (DDPG) in PyTorch.

# LunarLanderContinuous-v3 with Deep Deterministic Policy Gradient (DDPG)

This project implements **Deep Deterministic Policy Gradient (DDPG)** to solve the [LunarLanderContinuous-v3](https://www.gymlibrary.dev/environments/box2d/lunar_lander/) environment from OpenAI Gym.  


- **Environment**: LunarLanderContinuous-v3  
  - State space: 8-dimensional continuous state  
  - Action space: 2-dimensional continuous thrust values  
  - Reward: Encourages safe landing, penalizes crashes/fuel use  

- **Algorithm**: Deep Deterministic Policy Gradient (DDPG)  
  - Off-policy, actor–critic, model-free RL algorithm  
  - Well-suited for continuous control tasks  

- **Goal**: Train an agent to land the lunar module safely and efficiently.

Code structure:

├── ddpg_agent.py        # Core DDPG implementation (actor, critic, replay buffer, training loop)

├── train.py             # Training script

├── evaluate.py          # Evaluation script

├── utils.py             # Helper functions (plotting, saving models, etc.)

Extra Notes:
DDPG is effective for continuous control but sensitive to hyperparameters.
Replay buffer size, target network update rate (τ), and exploration noise strongly influence performance.
Extensions like TD3 or SAC could improve stability and robustness.
