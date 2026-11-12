"""Evaluate a DQN checkpoint and optionally record actual Pong gameplay."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from train import DEVICE, QNetwork, make_env


def play(checkpoint, episodes=5, seed=100, render=False, video=None,
         max_steps=27000, epsilon=0.0):
    """Return raw episode scores. Video captures one RGB frame per env step."""
    if episodes < 1 or max_steps < 1 or not 0 <= epsilon <= 1:
        raise ValueError("Positive episodes/max_steps and epsilon in [0, 1] required")
    if render and video:
        raise ValueError("Choose interactive rendering or video recording")
    env = make_env(render_mode="rgb_array" if video else "human" if render else None)
    writer = None
    results = []
    rng = np.random.default_rng(seed)
    try:
        model = QNetwork(env.action_space.n).to(DEVICE)
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
        model.eval()
        if video:
            path = Path(video)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(f"Video already exists: {path}")
        with torch.inference_mode():
            for episode in range(episodes):
                state, _ = env.reset(seed=seed + episode)
                total_reward = 0.0
                terminated = truncated = False
                if video and writer is None:
                    frame = env.render()
                    # Atari runs at 60 frames/s, with four frames per decision.
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             15.0, (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError("Could not open MP4 writer")
                for step in range(1, max_steps + 1):
                    if rng.random() < epsilon:
                        action = int(rng.integers(env.action_space.n))
                    else:
                        obs = torch.as_tensor(np.asarray(state), device=DEVICE).unsqueeze(0)
                        action = int(model(obs).argmax(dim=1).item())
                    state, reward, terminated, truncated, _ = env.step(action)
                    total_reward += float(reward)
                    if writer is not None:
                        writer.write(cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR))
                    if terminated or truncated:
                        break
                row = dict(episode=episode + 1, seed=seed + episode,
                           reward=total_reward, length=step,
                           terminated=bool(terminated), truncated=bool(truncated),
                           capped=not (terminated or truncated))
                results.append(row)
                print(json.dumps(row), flush=True)
    finally:
        if writer is not None:
            writer.release()
        env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=27000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    if args.results and args.results.exists():
        parser.error(f"Results file already exists: {args.results}")
    results = play(args.checkpoint, args.episodes, args.seed, args.render,
                   args.video, args.max_steps, args.epsilon)
    if args.results:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps({"checkpoint": str(args.checkpoint),
                                           "epsilon": args.epsilon,
                                           "episodes": results}, indent=2) + "\n")
    print(f"Mean raw reward: {np.mean([r['reward'] for r in results]):.2f}")


if __name__ == "__main__":
    main()
