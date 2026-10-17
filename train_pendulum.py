import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from ppo import (
    ContinuousActor,
    Critic,
    collect_continuous_rollout,
    compute_gae,
    normalise_advantages,
    update_continuous_ppo
)


def evaluate_policy(
    env_id,
    actor,
    seed,
    num_episodes
):
    """
    Evaluate the current policy without changing the actor.

    The Gaussian mean is used as the deterministic action so that
    repeated evaluations are easier to compare during this initial
    Pendulum learning check.
    """

    env = gym.make(env_id)

    episode_returns = []

    for episode in range(num_episodes):

        observation, info = env.reset(
            seed=seed + episode
        )

        episode_return = 0.0
        terminated = False
        truncated = False

        while not terminated and not truncated:

            observation_tensor = torch.as_tensor(
                observation,
                dtype=torch.float32
            )

            observation_tensor = observation_tensor.unsqueeze(0)

            with torch.no_grad():

                mean, std = actor(
                    observation_tensor
                )

                action_tensor = mean.squeeze(0)

            action_for_env = action_tensor.numpy()

            action_for_env = np.clip(
                action_for_env,
                env.action_space.low,
                env.action_space.high
            )

            observation, reward, terminated, truncated, info = env.step(
                action_for_env
            )

            episode_return = episode_return + reward

        episode_returns.append(
            episode_return
        )

    env.close()

    mean_return = np.mean(
        episode_returns
    )

    return mean_return


# Reproducibility
seed = 42
evaluation_seed = 1000

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# Environment
env_id = "Pendulum-v1"

env = gym.make(
    env_id
)

observation, info = env.reset(
    seed=seed
)

env.action_space.seed(
    seed
)

observation_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]


# Actor and critic

actor = ContinuousActor(
    observation_size,
    action_size
)

critic = Critic(
    observation_size
)


# PPO settings
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2

learning_rate = 3e-4
entropy_coefficient = 0.01
max_grad_norm = 0.5

num_steps = 256
minibatch_size = 64
update_epochs = 4

total_timesteps = 50_000
evaluation_episodes = 5
evaluation_frequency = 10


# Optimisers

actor_optimizer = torch.optim.Adam(
    actor.parameters(),
    lr=learning_rate
)

critic_optimizer = torch.optim.Adam(
    critic.parameters(),
    lr=learning_rate
)


# Number of complete rollout-update cycles
num_updates = total_timesteps // num_steps
actual_training_steps = num_updates * num_steps

print("Number of PPO updates:", num_updates)
print("Actual training steps:", actual_training_steps)


# Evaluate the untrained policy before learning
initial_mean_return = evaluate_policy(
    env_id,
    actor,
    evaluation_seed,
    evaluation_episodes
)

print()
print(
    "Initial mean return:",
    initial_mean_return
)


# Store evaluation results for the learning curve
evaluation_timesteps = [0]
evaluation_returns = [initial_mean_return]


# Training
for update in range(num_updates):

    # Collect fresh experience using the current policy
    rollout, observation = collect_continuous_rollout(
        env,
        actor,
        critic,
        observation,
        num_steps
    )

    # Calculate advantages and critic targets from this rollout
    advantages, returns = compute_gae(
        rollout,
        gamma,
        gae_lambda
    )

    # Normalise the advantages once before the PPO updates
    normalised_advantages = normalise_advantages(
        advantages
    )

    # Reuse this rollout for several PPO epochs
    update_continuous_ppo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        normalised_advantages,
        returns,
        clip_epsilon,
        entropy_coefficient,
        update_epochs,
        minibatch_size,
        max_grad_norm
    )

    timesteps_completed = (
        (update + 1) * num_steps
    )

    # Periodically check whether the policy is improving
    if (
        (update + 1) % evaluation_frequency == 0
        or update == num_updates - 1
    ):

        mean_return = evaluate_policy(
            env_id,
            actor,
            evaluation_seed,
            evaluation_episodes
        )

        evaluation_timesteps.append(
            timesteps_completed
        )

        evaluation_returns.append(
            mean_return
        )

        print()
        print(
            "Timesteps:",
            timesteps_completed
        )

        print(
            "Mean return:",
            mean_return
        )


env.close()