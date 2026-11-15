"""Play Pong with a trained DQN model (plain window fallback)."""

import argparse
from pathlib import Path

import numpy as np
import torch

from train import QNetwork, make_env


def main():
    """Play the requested number of full Pong games in the standard window."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    env = make_env(render_mode="human")
    model = QNetwork(env.action_space.n)
    checkpoint = Path(__file__).resolve().parent / "dqn_model.pth"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        score = 0
        terminated = truncated = False
        while not (terminated or truncated):
            with torch.no_grad():
                observation = torch.as_tensor(np.asarray(state)).unsqueeze(0)
                action = int(model(observation).argmax(dim=1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            score += int(reward)
        print(f"Episode {episode}: score {score:+d}")

    env.close()


if __name__ == "__main__":
    main()
