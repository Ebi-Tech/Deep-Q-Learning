"""Run Samuel's ten reproducible experiments and summarize measured results."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

# One change from baseline per run, except the explicitly labelled decay test.
EXPERIMENTS = [
    (1, 1e-4, .99, 32, 1., .01, 100000, "baseline"),
    (2, 2e-4, .99, 32, 1., .01, 100000, "higher learning rate"),
    (3, 2.5e-5, .99, 32, 1., .01, 100000, "lower learning rate"),
    (4, 1e-4, .97, 32, 1., .01, 100000, "lower discount"),
    (5, 1e-4, .995, 32, 1., .01, 100000, "higher discount"),
    (6, 1e-4, .99, 16, 1., .01, 100000, "smaller batch"),
    (7, 1e-4, .99, 64, 1., .01, 100000, "larger batch"),
    (8, 1e-4, .99, 32, 1., .05, 100000, "higher final epsilon"),
    (9, 1e-4, .99, 32, 1., .01, 20000, "faster exploration decay"),
    (10, 1e-4, .99, 32, .7, .01, 100000, "lower initial epsilon"),
]
PARAMETERS = ("learning_rate", "gamma", "batch_size", "epsilon_start",
              "epsilon_end", "epsilon_decay_steps")


def write_summary(repo, records, args):
    lines = ["# Member Name: Samuel", "",
             f"Budget: {args.steps:,} environment decisions per run; replay capacity "
             f"{args.buffer_size:,}; warm-up {args.learning_starts:,}; train every 4 "
             "decisions; target sync every 1,000. Training seed: 0.", "",
             f"Evaluation: {args.eval_episodes} greedy episodes per model, seeds starting "
             "at 100. Raw rewards are reported. No capped episodes are treated as wins.", "",
             "These single-seed pilot runs do not establish convergence or statistical "
             "superiority. Epsilon schedules longer than the budget do not reach their endpoint.", "",
             "| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |",
             "|---|---|---|---|---|---|---|---|"]
    for e in EXPERIMENTS:
        record = records.get(e[0])
        behaviour = "Not run"
        if record:
            behaviour = (f"{e[7]}; {record['episodes']} training episodes; last-10 raw "
                         f"return {record['last10_mean']:.2f}; evaluation mean "
                         f"{record['eval_mean']:.2f}, std {record['eval_std']:.2f}; "
                         f"{record['capped']} evaluation episodes capped")
        lines.append("| " + " | ".join(map(str, (*e[:7], behaviour))) + " |")
    lines += ["", "Machine-readable configurations and evaluation episodes are in "
              "`logs/Samuel/experiment_<n>.json`; raw training metrics are in the "
              "matching CSVs. Models are in `models/Samuel/`.", ""]
    (repo / 'SamuelHyperparameter_experiment_results.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=500000)
    parser.add_argument('--buffer_size', type=int, default=2000)
    parser.add_argument('--learning_starts', type=int, default=1000)
    parser.add_argument('--eval_episodes', type=int, default=5)
    args = parser.parse_args()
    if min(args.steps, args.buffer_size, args.learning_starts, args.eval_episodes) <= 0:
        parser.error('All budgets must be positive')
    repo = Path(__file__).resolve().parent
    for e in EXPERIMENTS:
        for file in (repo/'logs/Samuel'/f'experiment_{e[0]}.csv',
                     repo/'logs/Samuel'/f'experiment_{e[0]}.json',
                     repo/'models/Samuel'/f'experiment_{e[0]}.pth'):
            if file.exists():
                parser.error(f'Refusing to overwrite existing experiment: {file}')
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    records = {}
    for e in EXPERIMENTS:
        config = dict(zip(PARAMETERS, e[1:7]))
        command = [sys.executable, str(repo/'train.py'), '--member', 'Samuel',
                   '--experiment', str(e[0]), '--total_steps', str(args.steps),
                   '--buffer_size', str(args.buffer_size),
                   '--learning_starts', str(args.learning_starts)]
        for key, value in config.items():
            command += ['--'+key, str(value)]
        print(f'Running experiment {e[0]}: {e[7]}', flush=True)
        subprocess.run(command, cwd=repo, env=env, check=True)
        evaluation_file = repo/'logs/Samuel'/f'experiment_{e[0]}_evaluation.json'
        subprocess.run([sys.executable, str(repo/'play.py'),
                        str(repo/'models/Samuel'/f'experiment_{e[0]}.pth'),
                        '--episodes', str(args.eval_episodes), '--results', str(evaluation_file)],
                       cwd=repo, env=env, check=True)
        evaluation = json.loads(evaluation_file.read_text())['episodes']
        with (repo/'logs/Samuel'/f'experiment_{e[0]}.csv').open() as handle:
            episodes = list(csv.DictReader(handle))
        scores = [episode['reward'] for episode in evaluation]
        record = dict(experiment=e[0], config=config, budget=vars(args), training_seed=0,
                      episodes=len(episodes), evaluation=evaluation,
                      last10_mean=float(np.mean([float(r['reward']) for r in episodes[-10:]])) if episodes else None,
                      eval_mean=float(np.mean(scores)), eval_std=float(np.std(scores)),
                      capped=sum(r['capped'] for r in evaluation))
        # The documented budget should normally include finished episodes.
        if not episodes:
            raise RuntimeError('No training episodes completed; increase --steps')
        (repo/'logs/Samuel'/f'experiment_{e[0]}.json').write_text(json.dumps(record, indent=2)+'\n')
        records[e[0]] = record
        write_summary(repo, records, args)
    best = max(records.values(), key=lambda r: r['eval_mean'])
    print(f"Best measured mean: experiment {best['experiment']} ({best['eval_mean']:.2f}); ties use first run.")


if __name__ == '__main__':
    main()
