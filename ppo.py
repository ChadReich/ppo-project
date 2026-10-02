import torch
import torch.nn as nn


class Actor(nn.Module):

    def __init__(self, observation_size, action_size, hidden_size=64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, observation):

        logits = self.network(observation)

        return logits


class Critic(nn.Module):

    def __init__(self, observation_size, hidden_size=64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, observation):

        value = self.network(observation)

        # Remove the final size-1 dimension so values have shape [batch_size]
        value = value.squeeze(-1)

        return value