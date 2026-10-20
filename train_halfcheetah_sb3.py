# Imports
import csv
import os
import random

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import gymnasium as gym
import numpy as np
import torch


# Experimental settings
training_seeds = [0, 1, 2, 3, 4]
evaluation_seed = 1000

env_id = "HalfCheetah-v5"

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


# Create a matched evaluation function using the public SB3 API
def evaluate_model(
    model,
    env,
    evaluation_seed,
    num_episodes=5
):
    episode_returns = []

    # Loop through the specified number of episodes
    for episode in range(num_episodes):

        observation, info = env.reset(
            seed=evaluation_seed + episode
        )

        terminated = False
        truncated = False
        episode_return = 0.0

        # Loop until the episode is terminated or truncated
        while not (terminated or truncated):

            # Get the deterministic action from the model
            action, _ = model.predict(
                observation,
                deterministic=True
            )

            # Take a step in the environment using the action
            observation, reward, terminated, truncated, info = env.step(
                action
            )

            episode_return += reward

        episode_returns.append(
            episode_return
        )

    mean_return = float(
        np.mean(episode_returns)
    )

    return mean_return


# Store results from all training seeds
all_results = []
all_diagnostics = []

num_updates = total_timesteps // num_steps


# Train and evaluate each seed
for seed in training_seeds:

    print()
    print("Training seed:", seed)

    # Reproducibility
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

    # Evaluate the untrained model
    initial_evaluation_mean = evaluate_model(
        model,
        eval_env,
        evaluation_seed,
        num_episodes=num_evaluation_episodes
    )

    evaluation_means = [
        initial_evaluation_mean
    ]

    evaluation_steps = [
        0
    ]

    all_results.append(
        {
            "environment": env_id,
            "implementation": "sb3",
            "training_seed": seed,
            "timestep": 0,
            "mean_return": initial_evaluation_mean
        }
    )

    print(
        "Initial evaluation results (mean return):",
        initial_evaluation_mean
    )

    # Train one PPO rollout-update cycle at a time
    for update in range(num_updates):

        model.learn(
            total_timesteps=num_steps,
            reset_num_timesteps=False
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

        all_diagnostics.append(
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

            evaluation_means.append(
                mean_return
            )

            evaluation_steps.append(
                model.num_timesteps
            )

            all_results.append(
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

    # Close the environments after the seed is complete
    env.close()
    eval_env.close()

    # Save completed results after every seed
    os.makedirs(
        "results",
        exist_ok=True
    )

    with open(
        "results/halfcheetah_sb3.csv",
        "w",
        newline=""
    ) as csv_file:

        fieldnames = [
            "environment",
            "implementation",
            "training_seed",
            "timestep",
            "mean_return"
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(
            all_results
        )

    with open(
        "results/halfcheetah_sb3_diagnostics.csv",
        "w",
        newline=""
    ) as csv_file:

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

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(
            all_diagnostics
        )

    print()
    print("Finished seed:", seed)
    print("Results saved.")