"""Create training plots from the saved experiment logs."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_log(path):
    """Read episode data from one CSV log."""
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    steps = [int(row["step"]) for row in rows]
    rewards = [float(row["reward"]) for row in rows]
    lengths = [int(row["length"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    epsilons = [float(row["epsilon"]) for row in rows]
    return steps, rewards, lengths, losses, epsilons


def moving_average(values, window):
    """Return a simple moving average with its final value at each window."""
    return [sum(values[i - window + 1:i + 1]) / window
            for i in range(window - 1, len(values))]


def plot_best():
    """Save reward, length, and loss/epsilon plots for the selected run."""
    path = Path("logs/Samuel/experiment_8.csv")
    steps, rewards, lengths, losses, epsilons = read_log(path)
    episodes = list(range(1, len(rewards) + 1))
    average = moving_average(rewards, 20)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(episodes, rewards, alpha=0.3, label="Episode reward")
    ax.plot(episodes[19:], average, linewidth=2, label="20-episode average")
    ax.set(xlabel="Episode", ylabel="Reward", title="Samuel experiment 8 rewards")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("plots/best_episode_rewards.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(episodes, lengths)
    ax.set(xlabel="Episode", ylabel="Steps", title="Samuel experiment 8 episode lengths")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("plots/best_episode_lengths.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, losses, color="tab:blue", label="Training loss")
    ax.set(xlabel="Environment steps", ylabel="Loss")
    ax2 = ax.twinx()
    ax2.plot(steps, epsilons, color="tab:orange", label="Epsilon")
    ax2.set_ylabel("Epsilon")
    ax.set_title("Samuel experiment 8 loss and epsilon")
    ax.grid(alpha=0.3)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines])
    fig.tight_layout()
    fig.savefig("plots/best_loss_epsilon.png", dpi=150)
    plt.close(fig)


def plot_members():
    """Save one moving-average reward plot for each member."""
    for member in ("Divine", "Gabriel", "Samuel"):
        fig, ax = plt.subplots(figsize=(10, 6))
        for experiment in range(1, 11):
            path = Path(f"logs/{member}/experiment_{experiment}.csv")
            steps, rewards, _, _, _ = read_log(path)
            average = moving_average(rewards, 20)
            ax.plot(steps[19:], average, label=f"Experiment {experiment}")
        ax.set(xlabel="Environment steps", ylabel="20-episode average reward",
               title=f"{member} experiment comparison")
        ax.grid(alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(f"plots/{member.lower()}_comparison.png", dpi=150)
        plt.close(fig)


def main():
    """Create all requested plot files."""
    Path("plots").mkdir(exist_ok=True)
    plot_best()
    plot_members()
    for path in sorted(Path("plots").glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
