import csv
import os
import random

import gymnasium as gym
import numpy as np
import torch

from ppo import (
    ContinuousActor,
    Critic,
    collect_continuous_rollout,
    compute_gae,
    normalise_advantages,
    update_continuous_ppo,
    explained_variance
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
    policy performance can be compared consistently across checkpoints.
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

    mean_return = float(
        np.mean(episode_returns)
    )

    return mean_return


def save_results(
    results,
    filename
):
    """
    Save evaluation results to a CSV file.
    """

    os.makedirs(
        "results",
        exist_ok=True
    )

    filepath = os.path.join(
        "results",
        filename
    )

    fieldnames = [
        "environment",
        "implementation",
        "training_seed",
        "timestep",
        "mean_return"
    ]

    with open(
        filepath,
        "w",
        newline=""
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )

def save_diagnostics(
    diagnostics,
    filename
):
    """
    Save PPO diagnostic results to a CSV file.
    """

    os.makedirs(
        "results",
        exist_ok=True
    )

    filepath = os.path.join(
        "results",
        filename
    )

    fieldnames = [
        "environment",
        "implementation",
        "training_seed",
        "update",
        "timestep",
        "explained_variance",
        "approx_kl",
        "clip_fraction",
        "policy_std"
    ]

    with open(
        filepath,
        "w",
        newline=""
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            diagnostics
        )

# Experimental settings

env_id = "HalfCheetah-v5"

training_seeds = [0, 1, 2, 3, 4]

evaluation_seed = 1000
evaluation_episodes = 5

gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2

learning_rate = 3e-4
entropy_coefficient = 0.0
max_grad_norm = 0.5

num_steps = 2048
minibatch_size = 64
update_epochs = 10

total_timesteps = 999_424
evaluation_frequency = 16_384


# Store results from every training seed
all_results = []
all_diagnostics = []


# Run each training seed

for seed in training_seeds:

    print()
    print("-" * 60)
    print("Training seed:", seed)
    print("-" * 60)

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Environment
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

    print("Observation size:", observation_size)
    print("Action size:", action_size)

    # Actor and critic
    actor = ContinuousActor(
        observation_size,
        action_size
    )

    critic = Critic(
        observation_size
    )

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

    # Evaluate the untrained policy
    initial_mean_return = evaluate_policy(
        env_id,
        actor,
        evaluation_seed,
        evaluation_episodes
    )

    print()
    print("Initial mean return:", initial_mean_return)

    # Save timestep-zero evaluation
    all_results.append(
        {
            "environment": env_id,
            "implementation": "scratch",
            "training_seed": seed,
            "timestep": 0,
            "mean_return": initial_mean_return
        }
    )


    # Training
    for update in range(num_updates):

        # Collect fresh experience
        rollout, observation = collect_continuous_rollout(
            env,
            actor,
            critic,
            observation,
            num_steps
        )

        if update == 0:

            print()
            print("-" * 60)
            print("First HalfCheetah rollout shapes")
            print("Actions:", rollout["actions"].shape)
            print("Old log-probs:", rollout["old_log_probs"].shape)
            print("Values:", rollout["values"].shape)
            print("Next values:", rollout["next_values"].shape)

        # Calculate advantages and critic targets
        advantages, returns = compute_gae(
            rollout,
            gamma,
            gae_lambda
        )

        # Explained variance of the value function predictions
        explained_variance_value = explained_variance(
            returns,
            rollout["values"]
        )


        # Normalise advantages
        normalised_advantages = normalise_advantages(
            advantages
        )

        # PPO update
        mean_approx_kl, mean_clip_fraction, mean_policy_std = update_continuous_ppo(
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

        # Store diagnostics for this update
        all_diagnostics.append(
            {
                "environment": env_id,
                "implementation": "scratch",
                "training_seed": seed,
                "update": update + 1,
                "timestep": timesteps_completed,
                "explained_variance": explained_variance_value,
                "approx_kl": mean_approx_kl,
                "clip_fraction": mean_clip_fraction,
                "policy_std": mean_policy_std
            }
        )

        # Periodic deterministic evaluation
        if timesteps_completed % evaluation_frequency == 0:

            mean_return = evaluate_policy(
                env_id,
                actor,
                evaluation_seed,
                evaluation_episodes
            )

            print()
            print("Timesteps:", timesteps_completed)
            print("Mean return:", mean_return)

            all_results.append(
                {
                    "environment": env_id,
                    "implementation": "scratch",
                    "training_seed": seed,
                    "timestep": timesteps_completed,
                    "mean_return": mean_return
                }
            )

    env.close()

    # Save after every completed seed
    save_results(
        all_results,
        "halfcheetah_scratch.csv"
    )

    save_diagnostics(
        all_diagnostics,
        "halfcheetah_scratch_diagnostics.csv"
    )

    print()
    print("Finished seed:", seed)
    print("Results saved.")


print()
print("-" * 60)
print("All scratch PPO seeds complete.")
print("-" * 60)
