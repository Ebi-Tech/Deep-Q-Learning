# train.py
# Trains a Deep Q-Network (DQN) agent to play Atari Pong.
#
# Shared agreements between the three parts of this script:
#   - Each observation is a stack of 4 grayscale frames, shape (4, 84, 84), dtype uint8
#   - QNetwork(n_actions) takes a batch of observations and returns
#     Q-values of shape (batch_size, n_actions)
#   - All hyperparameters come from get_settings(), for example settings.learning_rate

import argparse

import ale_py
import gymnasium as gym
import torch

gym.register_envs(ale_py)

ENV_ID = "ALE/Pong-v5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_settings():
    """Read the hyperparameters from the command line, with default values."""
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari Pong")

    # Experiment identity: decides where the model and log are saved
    parser.add_argument("--member", type=str, required=True)
    parser.add_argument("--experiment", type=int, required=True)

    # Hyperparameters from the experiment table
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay_steps", type=int, default=100_000)

    # Other training settings, kept the same across all experiments
    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--buffer_size", type=int, default=50_000)
    parser.add_argument("--learning_starts", type=int, default=10_000)
    parser.add_argument("--train_every", type=int, default=4)
    parser.add_argument("--target_update_every", type=int, default=1_000)

    return parser.parse_args()


def make_env(render_mode=None):
    """Create Pong with the standard Atari preprocessing used for DQN."""
    # frameskip=1 here because AtariPreprocessing does the frame skipping itself
    env = gym.make(ENV_ID, frameskip=1, render_mode=render_mode)

    # Grayscale, resize to 84x84, repeat each action for 4 frames,
    # and up to 30 random no-op actions at the start of each episode
    env = gym.wrappers.AtariPreprocessing(
        env, frame_skip=4, screen_size=84, grayscale_obs=True, noop_max=30
    )

    # Stack the last 4 frames so the network can see motion (ball direction and speed)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env
