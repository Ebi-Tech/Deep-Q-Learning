"""Run a full Samuel baseline with evaluated checkpoints and live status files.

Uses the shared training loop unchanged. Run outside a GPU-restricted sandbox
for Apple MPS. Progress and final reports live inside the selected run folder.
"""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import torch
import train


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temporary, path)


def copy_checkpoint(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, suffix='.tmp')
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_id', required=True)
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--buffer_size', type=int, default=50_000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_starts', type=int, default=10_000)
    parser.add_argument('--save_every_episodes', type=int, default=25)
    parser.add_argument('--checkpoint_eval_episodes', type=int, default=5)
    parser.add_argument('--final_eval_episodes', type=int, default=20)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if not args.run_id.replace('_', '').replace('-', '').isalnum():
        parser.error('Run ID must contain letters, digits, underscores or hyphens')
    if min(args.steps, args.buffer_size, args.batch_size, args.learning_starts,
           args.save_every_episodes, args.checkpoint_eval_episodes,
           args.final_eval_episodes, args.threads) <= 0:
        parser.error('All numeric settings must be positive')
    if args.steps < args.learning_starts:
        parser.error('Training budget must reach learning_starts')
    repo = Path(__file__).resolve().parent
    os.chdir(repo)
    logs = Path('logs/Samuel') / args.run_id
    models = Path('models/Samuel') / args.run_id
    if logs.exists() or models.exists():
        parser.error('Run folder already exists; choose a fresh run ID')
    device = args.device
    if device == 'auto':
        device = ('cuda' if torch.cuda.is_available() else
                  'mps' if torch.backends.mps.is_available() else 'cpu')
    train.DEVICE = torch.device(device)
    # Import after selecting the device: play imports the shared DEVICE value.
    from play import play
    torch.set_num_threads(args.threads)
    logs.mkdir(parents=True)
    models.mkdir(parents=True)
    started = time.monotonic()
    recent = deque(maxlen=100)
    state = dict(phase='starting', pid=os.getpid(), device=device, step=0,
                 total_steps=args.steps, episode=0, best_checkpoint_mean=None,
                 best_checkpoint_step=None, latest_checkpoint=None)
    status_file = logs/'status.json'
    report = logs/'report.md'
    original_logger = train.Logger
    original_save = train.save_model
    old_argv = sys.argv[:]
    pending_checkpoint = False
    best_mean = -float('inf')
    best_path = models/'best.pth'
    evaluations = []

    def status(**updates):
        state.update(updates)
        elapsed = time.monotonic() - started
        speed = state['step'] / elapsed if elapsed else 0
        state.update(elapsed_seconds=elapsed, overall_steps_per_second=speed,
                     estimated_remaining_seconds=(args.steps-state['step'])/speed if speed else None)
        write_json(status_file, state)
        report.write_text(
            f"# Samuel full baseline: {args.run_id}\n\n"
            f"Phase: {state['phase']}. Device: {device}.\n\n"
            f"Progress: {state['step']:,} / {args.steps:,} decisions. "
            "Training interval: 1.\n\n"
            f"Latest 100-episode training mean: {state.get('training_mean_100')}.\n\n"
            f"Highest checkpoint selection mean: {state['best_checkpoint_mean']}.\n\n"
            f"Final independent evaluation mean: {state.get('heldout_mean')}.\n\n"
            "Checkpoint selection uses seeds starting at 100; final evaluation "
            "uses separate seeds starting at 1000. More steps do not guarantee "
            "a strong policy. Capped evaluations are explicitly recorded.\n")

    def evaluate(path, step):
        nonlocal best_mean
        status(phase='evaluating_checkpoint', latest_checkpoint=str(path))
        # Keep evaluation initialization from consuming training's Torch RNG.
        with torch.random.fork_rng(devices=[]):
            episodes = play(path, episodes=args.checkpoint_eval_episodes, seed=100)
        mean = float(np.mean([r['reward'] for r in episodes]))
        entry = dict(step=step, checkpoint=str(path), mean_reward=mean,
                     capped=sum(r['capped'] for r in episodes), episodes=episodes)
        evaluations.append(entry)
        write_json(logs/'checkpoint_evaluations.json', evaluations)
        if entry['capped'] == 0 and mean > best_mean:
            copy_checkpoint(path, best_path)
            best_mean = mean
            state.update(best_checkpoint_mean=mean, best_checkpoint_step=step)
        status(phase='training')

    class ProgressLogger(original_logger):
        def log_episode(self, episode, step, reward, length, epsilon, loss):
            nonlocal pending_checkpoint
            super().log_episode(episode, step, reward, length, epsilon, loss)
            recent.append(float(reward))
            pending_checkpoint = episode % args.save_every_episodes == 0
            status(phase='training', step=step, episode=episode, epsilon=epsilon,
                   latest_loss=loss, training_mean_100=float(np.mean(recent)))

    def save_with_evaluation(online, settings, filename=None):
        nonlocal pending_checkpoint
        path = original_save(online, settings, filename)
        state['latest_checkpoint'] = path
        if pending_checkpoint:
            pending_checkpoint = False
            snapshot = models / f"step_{state['step']:09d}.pth"
            copy_checkpoint(path, snapshot)
            evaluate(snapshot, state['step'])
        return path

    argv = ['train.py', '--member', f'Samuel/{args.run_id}', '--experiment', '11',
            '--total_steps', str(args.steps), '--train_every', '1',
            '--buffer_size', str(args.buffer_size), '--batch_size', str(args.batch_size),
            '--learning_starts', str(args.learning_starts),
            '--save_every_episodes', str(args.save_every_episodes),
            '--model_path', str(models/'latest.pth')]
    sys.argv = argv
    settings = train.get_settings()
    write_json(logs/'manifest.json', dict(settings=vars(settings), device=device,
               launcher=vars(args), torch=torch.__version__, training_seed=0,
               source_sha256={f: hashlib.sha256((repo/f).read_bytes()).hexdigest()
                              for f in ['train.py', 'play.py', 'run_long_samuel.py']}))
    try:
        train.Logger = ProgressLogger
        train.save_model = save_with_evaluation
        status(phase='training')
        train.main()
        state['step'] = args.steps
        final = models/f'final_{args.steps:09d}.pth'
        copy_checkpoint(models/'latest.pth', final)
        evaluate(final, args.steps)
        if not best_path.exists():
            raise RuntimeError('All checkpoint evaluations were capped; no valid selection')
        status(phase='final_evaluation')
        with torch.random.fork_rng(devices=[]):
            heldout = play(best_path, episodes=args.final_eval_episodes, seed=1000)
        mean = float(np.mean([r['reward'] for r in heldout]))
        write_json(logs/'final_evaluation.json', dict(checkpoint=str(best_path),
                   mean_reward=mean, std_reward=float(np.std([r['reward'] for r in heldout])),
                   episodes=heldout, capped=sum(r['capped'] for r in heldout)))
        status(phase='recording', heldout_mean=mean)
        videos = Path('videos/Samuel')/args.run_id
        videos.mkdir(parents=True, exist_ok=True)
        recorded = play(best_path, episodes=1, seed=1000, video=videos/'gameplay.mp4')
        write_json(videos/'gameplay.json', dict(checkpoint=str(best_path), episodes=recorded))
        if state['episode']:
            env = dict(os.environ, MPLCONFIGDIR=str(Path(tempfile.gettempdir())/'samuel-mpl'))
            subprocess.run([sys.executable, 'plot_results.py', str(logs/'experiment_11.csv'),
                            '--out_dir', str(Path('plots/Samuel')/args.run_id)], env=env, check=True)
        status(phase='completed', heldout_mean=mean)
    except BaseException as exc:
        status(phase='failed', error=f'{type(exc).__name__}: {exc}')
        (logs/'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        train.Logger = original_logger
        train.save_model = original_save
        sys.argv = old_argv


if __name__ == '__main__':
    main()
