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


# =====================================================================
# GABRIEL'S PART: replay buffer, epsilon decay, action selection, main loop
# =====================================================================
import random
import numpy as np


class ReplayBuffer:
    """Fixed-size circular buffer of (state, action, reward, next_state, done).

    States are stored as uint8 (0-255) to save memory. Once the buffer is full,
    the oldest experience is overwritten by the newest one.
    """

    def __init__(self, capacity, obs_shape=(4, 84, 84)):
        self.capacity = capacity
        self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0  # next slot to write
        self.size = 0      # number of valid experiences stored

    def push(self, state, action, reward, next_state, done):
        """Store one experience."""
        i = self.position
        self.states[i] = state
        self.next_states[i] = next_state
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Return a random batch as tensors on DEVICE.

        states / next_states: float32, shape (B, 4, 84, 84), values 0-255
        actions: int64 (B,)   rewards: float32 (B,)   dones: float32 (B,)
        """
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx], dtype=torch.float32, device=DEVICE),
            torch.as_tensor(self.actions[idx], device=DEVICE),
            torch.as_tensor(self.rewards[idx], device=DEVICE),
            torch.as_tensor(self.next_states[idx], dtype=torch.float32, device=DEVICE),
            torch.as_tensor(self.dones[idx], device=DEVICE),
        )

    def __len__(self):
        return self.size


def get_epsilon(step, settings):
    """Linear epsilon decay from epsilon_start to epsilon_end over epsilon_decay_steps."""
    fraction = min(step / settings.epsilon_decay_steps, 1.0)
    return settings.epsilon_start + fraction * (settings.epsilon_end - settings.epsilon_start)


def select_action(online_net, state, epsilon, n_actions):
    """Epsilon-greedy: random action with probability epsilon, else argmax Q."""
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        return int(online_net(state_t).argmax(dim=1).item())


# =====================================================================
# SAMUEL'S PART: training step, target sync, logging, saving.
# Public signatures are shared with Gabriel's main loop.
# =====================================================================
import os
import csv
import tempfile
from pathlib import Path
import torch.nn.functional as F


def train_step(online_net, target_net, optimizer, buffer, settings):
    """Update vanilla DQN and return scalar Huber loss.

    Replay supplies unnormalized float32 images and termination masks. The
    CNN normalizes pixels; time-limit truncation continues to bootstrap.
    Only the online network receives gradients. Scheduling belongs to main().
    """
    if settings.batch_size <= 0 or len(buffer) == 0:
        raise ValueError("Training requires a positive batch size and nonempty replay")
    online_net.train()
    target_net.eval()
    states, actions, rewards, next_states, dones = buffer.sample(settings.batch_size)
    q = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1).values
        target = rewards + settings.gamma * next_q * (1.0 - dones)
    loss = F.smooth_l1_loss(q, target)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite DQN loss; optimizer update skipped")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), 10.0, error_if_nonfinite=True)
    optimizer.step()
    return loss.item()


def sync_target(online_net, target_net):
    """Copy the online network weights into the target network."""
    target_net.load_state_dict(online_net.state_dict())
    target_net.requires_grad_(False)
    target_net.eval()


class Logger:
    """Writes one CSV row per finished episode to logs/<member>/experiment_<n>.csv."""

    fields = ("episode", "step", "reward", "length", "epsilon", "loss")

    def __init__(self, settings):
        folder = Path("logs") / settings.member
        folder.mkdir(parents=True, exist_ok=True)
        self.path = str(folder / f"experiment_{settings.experiment}.csv")
        # Refuse to silently destroy a previous experiment's evidence.
        with open(self.path, "x", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(self.fields)

    def log_episode(self, episode, step, reward, length, epsilon, loss):
        """Append raw return and the latest update loss, not an episode mean."""
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow((episode, step, reward, length, epsilon, loss))


def save_model(online_net, settings, filename=None):
    """Atomically save inference weights; return a path compatible with play.py.

    This is a weights checkpoint, not an exact training-resume snapshot.
    Custom destination directories are created automatically.
    """
    path = Path(filename) if filename is not None else (
        Path("models") / settings.member / f"experiment_{settings.experiment}.pth"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    os.close(descriptor)
    try:
        torch.save(online_net.state_dict(), temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(path)


# =====================================================================
# GABRIEL'S PART: main training loop
# =====================================================================
def main():
    settings = get_settings()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    env = make_env()
    n_actions = env.action_space.n

    online_net, target_net = create_networks(n_actions)
    optimizer = torch.optim.Adam(online_net.parameters(), lr=settings.learning_rate)

    buffer = ReplayBuffer(settings.buffer_size)
    logger = Logger(settings)

    print(f"Device: {DEVICE} | Member: {settings.member} | Experiment: {settings.experiment}")

    state, _ = env.reset(seed=0)
    state = np.asarray(state)
    episode, episode_reward, episode_length = 1, 0.0, 0
    last_loss = 0.0
    recent_rewards = []

    for step in range(1, settings.total_steps + 1):
        # 1. Choose an action (epsilon-greedy)
        epsilon = get_epsilon(step, settings)
        action = select_action(online_net, state, epsilon, n_actions)

        # 2. Act in the environment and store the experience
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.asarray(next_state)
        # Reward is clipped to [-1, 1] for stable learning; the logged reward stays raw
        buffer.push(state, action, np.sign(reward), next_state, terminated)

        state = next_state
        episode_reward += reward
        episode_length += 1

        # 3. Learn from a random batch
        if step >= settings.learning_starts and step % settings.train_every == 0:
            last_loss = train_step(online_net, target_net, optimizer, buffer, settings)

        # 4. Periodically copy online weights into the target network
        if step % settings.target_update_every == 0:
            sync_target(online_net, target_net)

        # 5. End of episode: log it and reset
        if terminated or truncated:
            logger.log_episode(episode, step, episode_reward, episode_length, epsilon, last_loss)
            recent_rewards.append(episode_reward)
            if episode % 10 == 0:
                avg = np.mean(recent_rewards[-10:])
                print(f"Episode {episode:4d} | step {step:7d} | avg reward (10) {avg:6.1f} "
                      f"| epsilon {epsilon:.3f} | loss {last_loss:.4f}")
            episode += 1
            episode_reward, episode_length = 0.0, 0
            state, _ = env.reset()
            state = np.asarray(state)

    path = save_model(online_net, settings)
    print(f"Training finished. Model saved to {path}")
    env.close()


if __name__ == "__main__":
    main()
