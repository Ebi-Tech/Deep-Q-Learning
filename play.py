"""Play Pong with a trained DQN model."""

import argparse
from pathlib import Path

import torch

from train import QNetwork, make_env


def main():
    """Play the requested number of full Pong games."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

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
                observation = torch.as_tensor(state).unsqueeze(0)
                action = int(model(observation).argmax(dim=1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
        print(f"Episode {episode}: score {score:.0f}")

    env.close()


if __name__ == "__main__":
    main()
