import argparse, json
from pathlib import Path
import numpy as np
import torch
import os
import torch
import train
requested = os.environ.get("SAMUEL_DEVICE", "auto")
chosen = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") if requested == "auto" else requested
if chosen == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable")
if chosen == "mps" and not torch.backends.mps.is_available(): raise RuntimeError("Apple MPS is unavailable")
train.DEVICE = torch.device(chosen)
torch.set_num_threads(int(os.environ.get("SAMUEL_THREADS", "2")))
print("Samuel device:", train.DEVICE, flush=True)


def evaluate(checkpoint, seeds, video=None, max_steps=27000):
    env = train.make_env(render_mode="rgb_array" if video else None)
    writer = None
    episodes = []
    try:
        model = train.QNetwork(env.action_space.n).to(train.DEVICE)
        model.load_state_dict(torch.load(checkpoint, map_location=train.DEVICE, weights_only=True))
        assert all(torch.isfinite(p).all() for p in model.parameters())
        model.eval()
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            total = 0.0
            terminated = truncated = False
            for step in range(1, max_steps + 1):
                if video:
                    import cv2
                    frame = env.render()
                    if writer is None:
                        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 15, (frame.shape[1], frame.shape[0]))
                        if not writer.isOpened(): raise RuntimeError("Cannot create MP4")
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                with torch.inference_mode():
                    q = model(torch.as_tensor(np.asarray(obs), device=train.DEVICE).unsqueeze(0))
                    assert torch.isfinite(q).all()
                    action = int(q.argmax(1).item())
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                if terminated or truncated: break
            episodes.append(dict(seed=seed, reward=total, length=step, terminated=bool(terminated), truncated=bool(truncated), capped=not (terminated or truncated)))
        rewards = [e["reward"] for e in episodes]
        return dict(checkpoint=str(checkpoint), mean_reward=float(np.mean(rewards)), std_reward=float(np.std(rewards)), wins=sum(e["reward"] > 0 and e["terminated"] and not e["truncated"] and not e["capped"] for e in episodes), episodes=episodes)
    finally:
        if writer is not None: writer.release()
        env.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint"); p.add_argument("result")
    p.add_argument("--episodes", type=int, default=5); p.add_argument("--seed", type=int, default=100)
    p.add_argument("--video")
    a = p.parse_args()
    if Path(a.result).exists() or (a.video and Path(a.video).exists()): raise FileExistsError("Evaluation outputs already exist")
    result = evaluate(a.checkpoint, range(a.seed, a.seed+a.episodes), a.video)
    Path(a.result).write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != "episodes"}, indent=2))
