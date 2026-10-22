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
    explained_variance,
    update_continuous_ppo,
)


def append_rows(path, fieldnames, rows):
    file_exists = os.path.exists(path)

    with open(path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)


def evaluate_policy(env_id, actor, evaluation_seed, num_episodes=5):
    env = gym.make(env_id)
    episode_returns = []

    for episode in range(num_episodes):
        observation, info = env.reset(seed=evaluation_seed + episode)

        terminated = False
        truncated = False
        episode_return = 0.0

        while not (terminated or truncated):
            observation_tensor = torch.as_tensor(
                observation,
                dtype=torch.float32,
            ).unsqueeze(0)

            with torch.no_grad():
                mean, std = actor(observation_tensor)
                action_tensor = mean.squeeze(0)

            action_for_env = action_tensor.numpy()
            action_for_env = np.clip(
                action_for_env,
                env.action_space.low,
                env.action_space.high,
            )

            observation, reward, terminated, truncated, info = env.step(
                action_for_env
            )

            episode_return += reward

        episode_returns.append(episode_return)

    env.close()

    return float(np.mean(episode_returns))


def main(seed):
    evaluation_seed = 1000
    env_id = "Pendulum-v1"

    total_timesteps = 999_424
    evaluation_frequency = 16_384
    num_evaluation_episodes = 5

    learning_rate = 3e-4
    num_steps = 2048
    minibatch_size = 64
    update_epochs = 10

    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    entropy_coefficient = 0.0
    max_grad_norm = 0.5

    num_updates = total_timesteps // num_steps

    print()
    print("Scratch PPO - training seed:", seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(env_id)
    observation, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    actor = ContinuousActor(observation_size, action_size)
    critic = Critic(observation_size)

    actor_optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=learning_rate,
    )

    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=learning_rate,
    )

    results = []
    diagnostics = []

    initial_evaluation_mean = evaluate_policy(
        env_id,
        actor,
        evaluation_seed,
        num_episodes=num_evaluation_episodes,
    )

    results.append(
        {
            "environment": env_id,
            "implementation": "scratch",
            "training_seed": seed,
            "timestep": 0,
            "mean_return": initial_evaluation_mean,
        }
    )

    print("Initial evaluation mean return:", initial_evaluation_mean)

    training_seconds = 0.0

    for update in range(num_updates):
        training_start = time.perf_counter()

        rollout, observation = collect_continuous_rollout(
            env,
            actor,
            critic,
            observation,
            num_steps,
        )

        advantages, returns = compute_gae(
            rollout,
            gamma,
            gae_lambda,
        )

        explained_variance_value = explained_variance(
            returns,
            rollout["values"],
        )

        normalised_advantages = normalise_advantages(advantages)

        mean_approx_kl, mean_clip_fraction, mean_policy_std = (
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
                max_grad_norm,
            )
        )

        training_seconds += time.perf_counter() - training_start

        timesteps_completed = (update + 1) * num_steps

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
                "policy_std": mean_policy_std,
            }
        )

        if timesteps_completed % evaluation_frequency == 0:
            mean_return = evaluate_policy(
                env_id,
                actor,
                evaluation_seed,
                num_episodes=num_evaluation_episodes,
            )

            results.append(
                {
                    "environment": env_id,
                    "implementation": "scratch",
                    "training_seed": seed,
                    "timestep": timesteps_completed,
                    "mean_return": mean_return,
                }
            )

            print(
                "Timesteps:",
                timesteps_completed,
                "Mean return:",
                mean_return,
            )

    env.close()

    interactions_per_second = total_timesteps / training_seconds
    seconds_per_update = training_seconds / num_updates

    runtime = [
        {
            "environment": env_id,
            "implementation": "scratch",
            "training_seed": seed,
            "training_timesteps": total_timesteps,
            "num_updates": num_updates,
            "training_seconds_excluding_evaluation": training_seconds,
            "interactions_per_second": interactions_per_second,
            "seconds_per_update": seconds_per_update,
        }
    ]

    os.makedirs("results", exist_ok=True)

    append_rows(
        "results/pendulum_scratch_frozen.csv",
        [
            "environment",
            "implementation",
            "training_seed",
            "timestep",
            "mean_return",
        ],
        results,
    )

    append_rows(
        "results/pendulum_scratch_frozen_diagnostics.csv",
        [
            "environment",
            "implementation",
            "training_seed",
            "update",
            "timestep",
            "explained_variance",
            "approx_kl",
            "clip_fraction",
            "policy_std",
        ],
        diagnostics,
    )

    append_rows(
        "results/pendulum_scratch_frozen_runtime.csv",
        [
            "environment",
            "implementation",
            "training_seed",
            "training_timesteps",
            "num_updates",
            "training_seconds_excluding_evaluation",
            "interactions_per_second",
            "seconds_per_update",
        ],
        runtime,
    )

    print()
    print("Finished scratch seed:", seed)
    print("Training seconds excluding evaluation:", training_seconds)
    print("Interactions per second:", interactions_per_second)
    print("Seconds per update:", seconds_per_update)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    main(args.seed)
