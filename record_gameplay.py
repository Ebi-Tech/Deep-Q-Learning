from pathlib import Path
import cv2, torch
from PIL import Image
from train import QNetwork, make_env

def main():
    """Record 30 seconds of greedy gameplay as MP4 or GIF."""
    env = make_env(render_mode="rgb_array")
    model = QNetwork(env.action_space.n)
    model.load_state_dict(torch.load("dqn_model.pth", map_location="cpu"))
    model.eval()
    state, _ = env.reset(seed=0)
    frame = env.render()
    writer = cv2.VideoWriter("plots/gameplay.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                             15.0, (frame.shape[1], frame.shape[0]))
    frames = []
    with torch.no_grad():
        for _ in range(450):
            action = int(model(torch.as_tensor(state).unsqueeze(0)).argmax(dim=1).item())
            state, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            frames.append(frame)
            if writer.isOpened():
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if terminated or truncated:
                state, _ = env.reset()
    writer.release()
    env.close()
    video = Path("plots/gameplay.mp4")
    fallback = not video.is_file() or video.stat().st_size == 0
    if fallback:
        video.unlink(missing_ok=True)
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save("plots/gameplay.gif", save_all=True, append_images=images[1:], duration=67, loop=0)
    print("plots/gameplay.gif" if fallback else "plots/gameplay.mp4")


if __name__ == "__main__":
    main()
