"""Plot full-run training/checkpoint returns beside independent test episodes."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_id', required=True)
    args = parser.parse_args()
    if not args.run_id.replace('_', '').replace('-', '').isalnum():
        parser.error('Invalid run ID')
    repo = Path(__file__).resolve().parent
    logs = repo/'logs/Samuel'/args.run_id
    with (logs/'experiment_11.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    checkpoints = json.loads((logs/'checkpoint_evaluations.json').read_text())
    evaluation = json.loads((logs/'final_evaluation.json').read_text())
    steps = [int(r['step']) for r in rows]
    rewards = [float(r['reward']) for r in rows]
    rolling = [np.mean(rewards[max(0, i-99):i+1]) for i in range(len(rewards))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    axes[0].plot(steps, rewards, alpha=.18, color='gray', label='Training episodes')
    axes[0].plot(steps, rolling, color='#3669a5', label='Training: trailing 100 episodes')
    axes[0].plot([r['step'] for r in checkpoints], [r['mean_reward'] for r in checkpoints],
                 'o-', color='#218557', markersize=4, label='Checkpoint selection: 5 greedy episodes')
    axes[0].set(title='Full-run learning curve', xlabel='Environment decisions', ylabel='Raw Pong return')
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000000:.1f}M'))
    axes[0].legend(fontsize=8, loc='upper left')
    scores = [r['reward'] for r in evaluation['episodes']]
    x = np.arange(1, len(scores)+1)
    axes[1].bar(x, scores, color=['#218557' if r > 0 else '#c65353' for r in scores])
    axes[1].axhline(evaluation['mean_reward'], color='#3669a5', linestyle='--',
                    label=f"Mean {evaluation['mean_reward']:+.2f}")
    axes[1].set(title='Independent evaluation of selected checkpoint',
                xlabel='Episode (separate seeds starting at 1000)', ylabel='Raw Pong return')
    axes[1].set_xticks(x)
    axes[1].legend()
    for ax in axes:
        ax.axhline(0, color='black', linewidth=.7)
        ax.set_ylim(-22, 22)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Samuel — 1,000,000-step baseline; one training seed; Apple MPS')
    fig.tight_layout()
    output = repo/'plots/Samuel'/args.run_id/'full_training_evaluation.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
