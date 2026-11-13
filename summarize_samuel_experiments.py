"""Validate and plot a completed Samuel run; optionally record its selected model."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from train import QNetwork


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_id', required=True)
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    if not args.run_id.replace('_', '').replace('-', '').isalnum():
        parser.error('Invalid run ID')
    repo = Path(__file__).resolve().parent
    os.chdir(repo)
    logs = Path('logs/Samuel') / args.run_id
    models = Path('models/Samuel') / args.run_id
    plots = Path('plots/Samuel') / args.run_id
    plots.mkdir(parents=True, exist_ok=True)
    records = []
    for n in range(1, 11):
        record = json.loads((logs/f'experiment_{n}.json').read_text())
        if record['experiment'] != n:
            raise ValueError(f'Experiment identity mismatch: {n}')
        with (logs/f'experiment_{n}.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        if not rows or len(rows) != record['episodes']:
            raise ValueError(f'Episode count mismatch: {n}')
        if not all(np.isfinite(float(r['loss'])) for r in rows):
            raise ValueError(f'Nonfinite loss: {n}')
        if sum(int(r['length']) for r in rows) != int(rows[-1]['step']):
            raise ValueError(f'Inconsistent cumulative steps: {n}')
        evaluation = record['evaluation']
        if len(evaluation) != record['budget']['eval_episodes']:
            raise ValueError(f'Evaluation count mismatch: {n}')
        if not np.isclose(np.mean([r['reward'] for r in evaluation]), record['eval_mean']):
            raise ValueError(f'Evaluation mean mismatch: {n}')
        path = models/f'experiment_{n}.pth'
        state = torch.load(path, map_location='cpu', weights_only=True)
        net = QNetwork(6)
        net.load_state_dict(state)
        net.eval()
        with torch.inference_mode():
            q_values = net(torch.full((1, 4, 84, 84), 255, dtype=torch.uint8))
        if not torch.isfinite(q_values).all():
            raise ValueError(f'Nonfinite inference output: {n}')
        if not all(torch.isfinite(t).all() for t in state.values()):
            raise ValueError(f'Nonfinite model parameters: {n}')
        record['model_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(record)
        subprocess.run([sys.executable, 'plot_results.py', str(logs/f'experiment_{n}.csv'),
                        '--out_dir', str(plots/f'experiment_{n}')], check=True)
    eligible = [r for r in records if r['capped'] == 0]
    if not eligible:
        raise ValueError('Every model has capped evaluations; increase evaluation budget')
    best = max(eligible, key=lambda r: r['eval_mean'])
    summary = dict(run_id=args.run_id, records=records,
                   total_training_steps=sum(r['budget']['steps'] for r in records),
                   total_evaluation_episodes=sum(len(r['evaluation']) for r in records),
                   capped_evaluation_episodes=sum(r['capped'] for r in records),
                   selected_experiment=best['experiment'], best_evaluation_mean=best['eval_mean'],
                   selection_note='Highest observed mean among runs without capped evaluations; first experiment breaks ties. Single-seed pilots do not establish superiority.')
    (logs/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(1, 11)
    axes[0].bar(x, [r['last10_mean'] for r in records], color='#4472aa')
    axes[0].set_title('Training: last 10 completed episodes')
    axes[1].bar(x, [r['eval_mean'] for r in records],
                yerr=[r['eval_std'] for r in records], capsize=4, color='#338966')
    axes[1].set_title('Greedy evaluation: mean ± std')
    for ax in axes:
        ax.set_xlabel('Experiment'); ax.set_ylabel('Raw Pong return')
        ax.set_xticks(x); ax.grid(axis='y', alpha=.25)
        ax.set_ylim(-22, max(0, max(r['eval_mean']+r['eval_std'] for r in records)+1))
    fig.suptitle(f"Samuel: {args.run_id} — {records[0]['budget']['steps']:,} decisions/run, one training seed")
    fig.tight_layout()
    fig.savefig(plots/'experiment_comparison.png', dpi=170)
    plt.close(fig)
    print(f"Validated 10 models; selected experiment {best['experiment']}, mean {best['eval_mean']:.2f}")
    if args.record:
        videos = Path('videos/Samuel') / args.run_id
        videos.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, 'play.py', str(models/f"experiment_{best['experiment']}.pth"),
                        '--episodes', '1', '--video', str(videos/'gameplay.mp4'),
                        '--results', str(videos/'gameplay.json')], check=True)


if __name__ == '__main__':
    main()
