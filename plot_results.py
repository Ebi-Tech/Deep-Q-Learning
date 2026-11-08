# plot_results.py
# Makes the training plots for the best run from its log CSV.
# Usage: uv run plot_results.py logs/Gabriel/experiment_10.csv
# Output: plots/episode_rewards.png, plots/episode_lengths.png,
#         plots/cumulative_steps.png, plots/training_summary.png

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def moving_average(values, window):
    window = max(1, min(window, len(values)))
    return np.convolve(values, np.ones(window) / window, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_csv")
    parser.add_argument("--out_dir", default="plots")
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    data = np.genfromtxt(args.log_csv, delimiter=",", names=True)
    episodes, steps = data["episode"], data["step"]
    rewards, lengths = data["reward"], data["length"]
    os.makedirs(args.out_dir, exist_ok=True)
    title = os.path.basename(args.log_csv)

    def single(fig_name, y, ylabel, label, smooth=True):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(episodes, y, alpha=0.35, label=label)
        if smooth and len(y) >= 2:
            ma = moving_average(y, args.window)
            ax.plot(episodes[len(episodes) - len(ma):], ma, linewidth=2,
                    label=f"{args.window}-episode average")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} per episode ({title})")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, fig_name), dpi=150)
        plt.close(fig)

    single("episode_rewards.png", rewards, "Reward", "Episode reward")
    single("episode_lengths.png", lengths, "Episode length (steps)", "Episode length")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(episodes, steps)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total environment steps")
    ax.set_title(f"Cumulative steps ({title})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "cumulative_steps.png"), dpi=150)
    plt.close(fig)

    # One combined figure for the README
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, y, lab in ((axes[0], rewards, "Reward"), (axes[1], lengths, "Episode length")):
        ax.plot(episodes, y, alpha=0.35)
        if len(y) >= 2:
            ma = moving_average(y, args.window)
            ax.plot(episodes[len(episodes) - len(ma):], ma, linewidth=2)
        ax.set_xlabel("Episode"); ax.set_ylabel(lab); ax.set_title(lab); ax.grid(alpha=0.3)
    axes[2].plot(episodes, steps)
    axes[2].set_xlabel("Episode"); axes[2].set_ylabel("Total steps")
    axes[2].set_title("Cumulative steps"); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "training_summary.png"), dpi=150)
    plt.close(fig)
    print(f"Saved plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
