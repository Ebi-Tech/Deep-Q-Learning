"""Run a full Samuel baseline with evaluated checkpoints and live status files.

Reuses the shared environment, replay, action selection and training functions. Run outside a GPU-restricted sandbox
for Apple MPS. Progress and final reports live inside the selected run folder.
"""
import argparse
from collections import deque
import hashlib
import json
import os
import random
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


def get_args():
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
    return args


def baseline_settings(args, models):
    """Keep the full baseline configuration explicit and reproducible."""
    return argparse.Namespace(
        member=f'Samuel/{args.run_id}', experiment=11,
        model_path=str(models/'latest.pth'), total_steps=args.steps,
        learning_rate=1e-4, gamma=.99, batch_size=args.batch_size,
        epsilon_start=1.0, epsilon_end=.01, epsilon_decay_steps=100000,
        buffer_size=args.buffer_size, learning_starts=args.learning_starts,
        train_every=1, target_update_every=1000,
        save_every_episodes=args.save_every_episodes)


def publish_status(logs, state, started):
    elapsed = time.monotonic() - started
    speed = state['step'] / elapsed if elapsed else 0
    state.update(elapsed_seconds=elapsed, overall_steps_per_second=speed,
                 estimated_remaining_seconds=(state['total_steps']-state['step'])/speed if speed else None)
    write_json(logs/'status.json', state)
    (logs/'report.md').write_text(
        f"# Samuel full baseline: {logs.name}\n\n"
        f"Phase: {state['phase']}. Device: {state['device']}.\n\n"
        f"Progress: {state['step']:,} / {state['total_steps']:,} decisions. Training interval: 1.\n\n"
        f"Latest 100-episode training mean: {state.get('training_mean_100')}.\n\n"
        f"Highest checkpoint selection mean: {state['best_checkpoint_mean']}.\n\n"
        f"Final independent evaluation mean: {state.get('heldout_mean')}.\n\n"
        "Selection uses seeds starting at 100; independent evaluation starts at 1000.\n")


def evaluate_snapshot(path, step, args, logs, models, state, started):
    from play import play
    state.update(phase='evaluating_checkpoint', latest_checkpoint=str(path))
    publish_status(logs, state, started)
    # Evaluation must not consume the training network's initialization RNG.
    with torch.random.fork_rng(devices=[]):
        episodes = play(path, episodes=args.checkpoint_eval_episodes, seed=100)
    mean = float(np.mean([row['reward'] for row in episodes]))
    entry = dict(step=step, checkpoint=str(path), mean_reward=mean,
                 capped=sum(row['capped'] for row in episodes), episodes=episodes)
    history_file = logs/'checkpoint_evaluations.json'
    history = json.loads(history_file.read_text()) if history_file.exists() else []
    write_json(history_file, history + [entry])
    best = state['best_checkpoint_mean']
    if entry['capped'] == 0 and (best is None or mean > best):
        copy_checkpoint(path, models/'best.pth')
        state.update(best_checkpoint_mean=mean, best_checkpoint_step=step)
    state['phase'] = 'training'
    publish_status(logs, state, started)


def train_agent(settings, args, logs, models, state, started):
    """Explicit baseline loop using Divine's and Gabriel's existing interfaces."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env = train.make_env()
    try:
        online, target = train.create_networks(env.action_space.n)
        optimizer = torch.optim.Adam(online.parameters(), lr=settings.learning_rate, eps=1.5e-4)
        buffer = train.ReplayBuffer(settings.buffer_size)
        logger = train.Logger(settings)
        observation, _ = env.reset(seed=0)
        observation = np.asarray(observation)
        episode, reward_sum, length, loss = 1, 0., 0, 0.
        recent = deque(maxlen=100)
        for step in range(1, settings.total_steps + 1):
            epsilon = train.get_epsilon(step, settings)
            action = train.select_action(online, observation, epsilon, env.action_space.n)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_observation = np.asarray(next_observation)
            buffer.push(observation, action, reward, next_observation, terminated)
            observation = next_observation
            reward_sum += reward
            length += 1
            if step >= settings.learning_starts:
                loss = train.train_step(online, target, optimizer,
                                        buffer.sample(settings.batch_size), settings.gamma)
            if step % settings.target_update_every == 0:
                train.sync_target(online, target)
            if terminated or truncated:
                logger.log_episode(episode, step, reward_sum, length, epsilon, loss)
                recent.append(float(reward_sum))
                state.update(step=step, episode=episode, epsilon=epsilon, latest_loss=loss,
                             training_mean_100=float(np.mean(recent)))
                publish_status(logs, state, started)
                if episode % settings.save_every_episodes == 0:
                    latest = train.save_model(online, settings)
                    snapshot = models/f'step_{step:09d}.pth'
                    copy_checkpoint(latest, snapshot)
                    evaluate_snapshot(snapshot, step, args, logs, models, state, started)
                episode += 1
                reward_sum, length = 0., 0
                observation, _ = env.reset()
                observation = np.asarray(observation)
        train.save_model(online, settings)
    finally:
        env.close()


def finish_run(args, logs, models, state, started):
    from play import play
    state['step'] = args.steps
    final = models/f'final_{args.steps:09d}.pth'
    copy_checkpoint(models/'latest.pth', final)
    evaluate_snapshot(final, args.steps, args, logs, models, state, started)
    best = models/'best.pth'
    if not best.exists():
        raise RuntimeError('All checkpoint evaluations were capped; no valid selection')
    state['phase'] = 'final_evaluation'
    publish_status(logs, state, started)
    with torch.random.fork_rng(devices=[]):
        heldout = play(best, episodes=args.final_eval_episodes, seed=1000)
    scores = [row['reward'] for row in heldout]
    write_json(logs/'final_evaluation.json', dict(checkpoint=str(best),
               mean_reward=float(np.mean(scores)), std_reward=float(np.std(scores)),
               episodes=heldout, capped=sum(row['capped'] for row in heldout)))
    state.update(phase='recording', heldout_mean=float(np.mean(scores)))
    publish_status(logs, state, started)
    videos = Path('videos/Samuel')/args.run_id
    videos.mkdir(parents=True, exist_ok=True)
    recorded = play(best, episodes=1, seed=1000, video=videos/'gameplay.mp4')
    write_json(videos/'gameplay.json', dict(checkpoint=str(best), episodes=recorded))
    if state['episode']:
        env = dict(os.environ, MPLCONFIGDIR=str(Path(tempfile.gettempdir())/'samuel-mpl'))
        subprocess.run([sys.executable, 'plot_results.py', str(logs/'experiment_11.csv'),
                        '--out_dir', str(Path('plots/Samuel')/args.run_id)], env=env, check=True)
    state['phase'] = 'completed'
    publish_status(logs, state, started)


def main():
    args = get_args()
    repo = Path(__file__).resolve().parent
    os.chdir(repo)
    logs, models = Path('logs/Samuel')/args.run_id, Path('models/Samuel')/args.run_id
    if logs.exists() or models.exists():
        raise FileExistsError('Run folder already exists; choose a fresh run ID')
    device = args.device
    if device == 'auto':
        device = ('cuda' if torch.cuda.is_available() else
                  'mps' if torch.backends.mps.is_available() else 'cpu')
    train.DEVICE = torch.device(device)
    torch.set_num_threads(args.threads)
    logs.mkdir(parents=True)
    models.mkdir(parents=True)
    settings = baseline_settings(args, models)
    state = dict(phase='training', pid=os.getpid(), device=device, step=0,
                 total_steps=args.steps, episode=0, best_checkpoint_mean=None,
                 best_checkpoint_step=None, latest_checkpoint=None)
    started = time.monotonic()
    write_json(logs/'manifest.json', dict(settings=vars(settings), device=device,
               launcher=vars(args), torch=torch.__version__, training_seed=0,
               source_sha256={file: hashlib.sha256((repo/file).read_bytes()).hexdigest()
                              for file in ['train.py', 'play.py', 'run_long_samuel.py']}))
    try:
        publish_status(logs, state, started)
        train_agent(settings, args, logs, models, state, started)
        finish_run(args, logs, models, state, started)
    except BaseException as exc:
        state.update(phase='failed', error=f'{type(exc).__name__}: {exc}')
        publish_status(logs, state, started)
        (logs/'error.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
