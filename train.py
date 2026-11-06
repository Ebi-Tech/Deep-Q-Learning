# train.py
# Trains a Deep Q-Network (DQN) agent to play Atari Pong.
#
# Shared agreements between the three parts of this script:
#   - Each observation is a stack of 4 grayscale frames, shape (4, 84, 84), dtype uint8
#   - QNetwork(n_actions) takes a batch of observations and returns
#     Q-values of shape (batch_size, n_actions)
#   - create_networks(n_actions) returns (online_net, target_net), both already on DEVICE
#   - All hyperparameters come from get_settings(), for example settings.learning_rate

import argparse

import ale_py
import gymnasium as gym
import torch
import torch.nn as nn

gym.register_envs(ale_py)

ENV_ID = "ALE/Pong-v5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_settings():
    """Read the hyperparameters from the command line, with default values.

    --member and --experiment decide where the model and log are saved.
    The next six are the hyperparameters from the experiment table.
    The rest are training settings kept the same across all experiments.
    """
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari Pong")

    parser.add_argument("--member", type=str, required=True)
    parser.add_argument("--experiment", type=int, required=True)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay_steps", type=int, default=100_000)

    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--buffer_size", type=int, default=50_000)
    parser.add_argument("--learning_starts", type=int, default=10_000)
    parser.add_argument("--train_every", type=int, default=4)
    parser.add_argument("--target_update_every", type=int, default=1_000)

    return parser.parse_args()


def make_env(render_mode=None):
    """Create Pong with the standard Atari preprocessing used for DQN.

    frameskip=1 is passed to gym.make because AtariPreprocessing does the
    frame skipping itself. AtariPreprocessing converts frames to grayscale,
    resizes them to 84x84, repeats each action for 4 frames, and does up to
    30 random no-op actions at the start of each episode.
    FrameStackObservation then stacks the last 4 frames so the network can
    see motion, like the direction and speed of the ball.
    """
    env = gym.make(ENV_ID, frameskip=1, render_mode=render_mode)
    env = gym.wrappers.AtariPreprocessing(
        env, frame_skip=4, screen_size=84, grayscale_obs=True, noop_max=30
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env


class QNetwork(nn.Module):
    """CNN that takes a stack of 4 frames and outputs one Q-value per action.

    The convolutional layers find visual features (paddles, ball, position):
    (4, 84, 84) -> (32, 20, 20) -> (64, 9, 9) -> (64, 7, 7), flattened to 3136.
    The fully connected layers turn those features into Q-values.
    Frames are stored as uint8 (0 to 255), so forward() scales them to 0 to 1 first.
    """

    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, obs):
        x = obs.float() / 255.0
        return self.fc(self.conv(x))


def create_networks(n_actions):
    """Create the online network and the target network, both on DEVICE.

    The online network is the one we train. The target network starts as
    an exact copy of it and is only used to compute the Bellman targets.
    It is never trained directly: it gets updated by copying the online
    weights into it every few steps. eval() marks it as not being trained.
    """
    online_net = QNetwork(n_actions).to(DEVICE)
    target_net = QNetwork(n_actions).to(DEVICE)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()
    return online_net, target_net
