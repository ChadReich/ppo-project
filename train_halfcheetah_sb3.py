# Imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import gymnasium as gym
import numpy as np

# Reproducibility
seed = 42
evaluation_seed = 1000

# Environment
env_id = "HalfCheetah-v5"


# Create HalfCheetah training environment
env = make_vec_env(env_id, n_envs=1, seed=seed)

# Separate evaluation environment for reproducibility
eval_env = gym.make(env_id)

# Create the SB3 PPO model
model = PPO("MlpPolicy", env, seed=seed, verbose=1)

# Create a matched evaluation function using the public SB3 API
def evaluate_model(model, env, evaluation_seed, num_episodes=5):
    episode_returns = []

    # Loop through the specified number of episodes
    for episode in range(num_episodes):
        observation, info = env.reset(seed=evaluation_seed + episode)
        terminated = False
        truncated = False
        episode_return = 0.0

        # Loop until the episode is terminated or truncated
        while not (terminated or truncated):
            # Get the action from the model
            action, _ = model.predict(observation, deterministic=True)

            # Take a step in the environment using the action
            observation, reward, terminated, truncated, info = env.step(action)

            episode_return += reward

        episode_returns.append(episode_return)

    mean_return = np.mean(episode_returns)

    return mean_return

# Evaluate the untrained model
initial_evaluation_mean = evaluate_model(model, eval_env, evaluation_seed, num_episodes=5)

# Train the model
model.learn(total_timesteps=49_152)

# Evaluate the trained model
trained_evaluation_mean = evaluate_model(model, eval_env, evaluation_seed, num_episodes=5)

print("Initial evaluation results (mean return):", initial_evaluation_mean)
print("Trained evaluation results (mean return):", trained_evaluation_mean)



# Close the environments
env.close()
eval_env.close()


