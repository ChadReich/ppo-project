import argparse
import csv
import os
import random
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import gymnasium as gym
import numpy as np
import torch


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


def evaluate_model(
    model,
    env,
    evaluation_seed,
    num_episodes=5
):
    """
    Evaluate the current policy using deterministic actions.
    """

    episode_returns = []

    for episode in range(num_episodes):

        observation, info = env.reset(
            seed=evaluation_seed + episode
        )

        terminated = False
        truncated = False
        episode_return = 0.0

        while not (terminated or truncated):

            action, _ = model.predict(
                observation,
                deterministic=True
            )

            observation, reward, terminated, truncated, info = env.step(
                action
            )

            episode_return = episode_return + reward

        episode_returns.append(
            episode_return
        )

    mean_return = float(
        np.mean(episode_returns)
    )

    return mean_return


def main(seed):

    # Experimental settings

    env_id = "HalfCheetah-v5"

    evaluation_seed = 1000
    num_evaluation_episodes = 5

    total_timesteps = 999_424
    evaluation_frequency = 16_384

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

    # Reproducibility

    print()
    print("SB3 PPO - training seed:", seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create HalfCheetah training environment

    env = make_vec_env(
        env_id,
        n_envs=1,
        seed=seed
    )

    # Separate evaluation environment for reproducibility

    eval_env = gym.make(
        env_id
    )

    # Create the SB3 PPO model using the matched PPO settings

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=num_steps,
        batch_size=minibatch_size,
        n_epochs=update_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_epsilon,
        ent_coef=entropy_coefficient,
        max_grad_norm=max_grad_norm,
        seed=seed,
        verbose=0
    )

    results = []
    diagnostics = []

    # Evaluate the untrained model

    initial_evaluation_mean = evaluate_model(
        model,
        eval_env,
        evaluation_seed,
        num_episodes=num_evaluation_episodes
    )

    results.append(
        {
            "environment": env_id,
            "implementation": "sb3",
            "training_seed": seed,
            "timestep": 0,
            "mean_return": initial_evaluation_mean
        }
    )

    print(
        "Initial evaluation mean return:",
        initial_evaluation_mean
    )

    # Training time excludes deterministic evaluation

    training_seconds = 0.0

    for update in range(num_updates):

        training_start = time.perf_counter()

        model.learn(
            total_timesteps=num_steps,
            reset_num_timesteps=False
        )

        training_seconds = training_seconds + (
            time.perf_counter() - training_start
        )

        # Read the diagnostics calculated internally by SB3

        logger_values = model.logger.name_to_value

        explained_variance_value = float(
            logger_values["train/explained_variance"]
        )

        mean_approx_kl = float(
            logger_values["train/approx_kl"]
        )

        mean_clip_fraction = float(
            logger_values["train/clip_fraction"]
        )

        mean_policy_std = float(
            logger_values["train/std"]
        )

        diagnostics.append(
            {
                "environment": env_id,
                "implementation": "sb3",
                "training_seed": seed,
                "update": update + 1,
                "timestep": model.num_timesteps,
                "explained_variance": explained_variance_value,
                "approx_kl": mean_approx_kl,
                "clip_fraction": mean_clip_fraction,
                "policy_std": mean_policy_std
            }
        )

        # Periodic deterministic evaluation

        if model.num_timesteps % evaluation_frequency == 0:

            mean_return = evaluate_model(
                model,
                eval_env,
                evaluation_seed,
                num_episodes=num_evaluation_episodes
            )

            results.append(
                {
                    "environment": env_id,
                    "implementation": "sb3",
                    "training_seed": seed,
                    "timestep": model.num_timesteps,
                    "mean_return": mean_return
                }
            )

            print(
                "Timesteps:",
                model.num_timesteps,
                "Mean return:",
                mean_return
            )

    env.close()
    eval_env.close()

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
            "implementation": "sb3",
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
        "results/halfcheetah_sb3_frozen.csv",
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
        "results/halfcheetah_sb3_frozen_diagnostics.csv",
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
        "results/halfcheetah_sb3_frozen_runtime.csv",
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
    print("Finished SB3 seed:", seed)
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
