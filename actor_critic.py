import torch.nn as nn
import torch.nn.functional as F
import torch


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1=400, hidden_dim2=300):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, action_dim)

        # # initialize weights
        #  self.apply(lambda m: init_weights(m, 1.0 / (2 * hidden1)**0.5))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # for continuous action spaces
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1=400, hidden_dim2=300):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # no activation for value output
        return x
