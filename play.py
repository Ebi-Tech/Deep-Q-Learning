"""Play Pong and show the game next to what the agent actually sees."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from train import QNetwork, make_env

SCALE = 3          # how much to enlarge the game frame
FRAME_DELAY = 30   # milliseconds between frames, controls playback speed


def build_view(game_rgb, observation, episode, score):
    """Place the full game beside the 84x84 grayscale image the network sees."""
    game = cv2.resize(cv2.cvtColor(game_rgb, cv2.COLOR_RGB2BGR),
                      (game_rgb.shape[1] * SCALE, game_rgb.shape[0] * SCALE),
                      interpolation=cv2.INTER_NEAREST)
    height = game.shape[0]

    agent_frame = np.asarray(observation)[-1]
    agent = cv2.resize(cv2.cvtColor(agent_frame, cv2.COLOR_GRAY2BGR),
                       (height, height), interpolation=cv2.INTER_NEAREST)

    gap = np.zeros((height, 20, 3), dtype=np.uint8)
    combined = np.hstack([game, gap, agent])

    cv2.putText(combined, f"Game    Score {score:+d}    Episode {episode}",
                (15, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combined, "What the agent sees (84x84)",
                (game.shape[1] + 30, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    return combined


def main():
    """Play the requested number of greedy games in a window."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    env = make_env(render_mode="rgb_array")
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
            cv2.imshow("DQN Pong", build_view(env.render(), state, episode, score))
            if cv2.waitKey(FRAME_DELAY) == ord("q"):
                env.close()
                cv2.destroyAllWindows()
                return
        print(f"Episode {episode}: score {score:+d}")

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
