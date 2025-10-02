import random
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device = torch.device("cpu")):
        # deque(maxlen=capacity) for automatic FIFO overwrite
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.as_tensor(np.array(states), device=self.device, dtype=torch.float32)
        actions = torch.as_tensor(np.array(actions), device=self.device, dtype=torch.float32)
        rewards = torch.as_tensor(np.array(rewards)[:, None], device=self.device, dtype=torch.float32)
        next_states = torch.as_tensor(np.array(next_states), device=self.device, dtype=torch.float32)
        dones = torch.as_tensor(np.array(dones)[:, None], device=self.device, dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return current number of stored transitions."""
        return len(self.buffer)





