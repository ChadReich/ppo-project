import argparse
import csv
import os
import random
import time

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


def append_rows(
    path,
    fieldnames,
    rows
):
    """
    Append completed seed results to a CSV file.
    """

    file_exists = os.path.exists(
        path
    )

    with open(
        path,
        "a",
        newline=""
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(
            rows
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

    env = gym.make(
        env_id
    )

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


def main(seed):

    # Experimental settings

    env_id = "HalfCheetah-v5"

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

    # Reproducibility

    print()
    print("Scratch PPO - training seed:", seed)

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

    results = []
    diagnostics = []

    # Evaluate the untrained policy

    initial_mean_return = evaluate_policy(
        env_id,
        actor,
        evaluation_seed,
        evaluation_episodes
    )

    results.append(
        {
            "environment": env_id,
            "implementation": "scratch",
            "training_seed": seed,
            "timestep": 0,
            "mean_return": initial_mean_return
        }
    )

    print("Initial mean return:", initial_mean_return)

    # Training time excludes deterministic evaluation

    training_seconds = 0.0

    for update in range(num_updates):

        training_start = time.perf_counter()

        # Collect fresh experience

        rollout, observation = collect_continuous_rollout(
            env,
            actor,
            critic,
            observation,
            num_steps
        )

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

        training_seconds = training_seconds + (
            time.perf_counter() - training_start
        )

        timesteps_completed = (
            (update + 1) * num_steps
        )

        # Store diagnostics for this update

        diagnostics.append(
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

            results.append(
                {
                    "environment": env_id,
                    "implementation": "scratch",
                    "training_seed": seed,
                    "timestep": timesteps_completed,
                    "mean_return": mean_return
                }
            )

            print(
                "Timesteps:",
                timesteps_completed,
                "Mean return:",
                mean_return
            )

    env.close()

    # Runtime efficiency metrics

    interactions_per_second = (
        total_timesteps / training_seconds
    )

    seconds_per_update = (
        training_seconds / num_updates
    )

    runtime = [
        {
            "environment": env_id,
            "implementation": "scratch",
            "training_seed": seed,
            "training_timesteps": total_timesteps,
            "num_updates": num_updates,
            "training_seconds_excluding_evaluation": training_seconds,
            "interactions_per_second": interactions_per_second,
            "seconds_per_update": seconds_per_update
        }
    ]

    os.makedirs(
        "results",
        exist_ok=True
    )

    append_rows(
        "results/halfcheetah_scratch_frozen.csv",
        [
            "environment",
            "implementation",
            "training_seed",
            "timestep",
            "mean_return"
        ],
        results
    )

    append_rows(
        "results/halfcheetah_scratch_frozen_diagnostics.csv",
        [
            "environment",
            "implementation",
            "training_seed",
            "update",
            "timestep",
            "explained_variance",
            "approx_kl",
            "clip_fraction",
            "policy_std"
        ],
        diagnostics
    )

    append_rows(
        "results/halfcheetah_scratch_frozen_runtime.csv",
        [
            "environment",
            "implementation",
            "training_seed",
            "training_timesteps",
            "num_updates",
            "training_seconds_excluding_evaluation",
            "interactions_per_second",
            "seconds_per_update"
        ],
        runtime
    )

    print()
    print("Finished scratch seed:", seed)
    print("Training seconds excluding evaluation:", training_seconds)
    print("Interactions per second:", interactions_per_second)
    print("Seconds per update:", seconds_per_update)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        required=True
    )

    args = parser.parse_args()

    main(
        args.seed
    )
