import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

env = gym.make("HalfCheetah-v5")

# Print the observation space and action space
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

# Reset the environment and get the initial observation
observation, info = env.reset(seed=42)
print("Initial observation shape:", observation.shape)

# Sample a random action from the action space
action = env.action_space.sample()

# Print the sampled action and its shape
print("Sampled action:", action)
print("Action shape:", action.shape)

# Take a step in the environment using the sampled action
next_observation, reward, terminated, truncated, info = env.step(
    action
)

print("Next observation shape:", next_observation.shape)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)

env.close()
