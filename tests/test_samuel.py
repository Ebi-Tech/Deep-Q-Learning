"""Regression checks for Samuel's training and persistence contracts."""
import csv
import contextlib
import io
import os
import subprocess
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from train import Logger, get_settings, save_model, sync_target, train_step
import play as playback


class Batch:
    def __len__(self):
        return 2

    def sample(self, batch_size):
        assert batch_size == 2
        # A terminal transition must ignore the target value of 10.
        return (torch.ones(2, 1), torch.tensor([0, 1]), torch.tensor([1., 2.]),
                torch.ones(2, 1), torch.tensor([1., 0.]))


class SamuelTests(unittest.TestCase):
    def test_default_playback_loads_submission_model_greedily_and_renders(self):
        with patch('sys.argv', ['play.py']), contextlib.redirect_stdout(io.StringIO()):
            with patch.object(playback, 'play', return_value=[{'reward': 1.}]) as run:
                playback.main()
        args = run.call_args.args
        self.assertEqual(args[0], Path(playback.__file__).resolve().parent/'dqn_model.pth')
        self.assertEqual(args[1], 1)
        self.assertTrue(args[3])
        self.assertEqual(args[6], 0.)

    def test_headless_evaluation_does_not_open_display(self):
        with patch('sys.argv', ['play.py', '--headless', '--episodes', '2']):
            with contextlib.redirect_stdout(io.StringIO()):
                with patch.object(playback, 'play', return_value=[{'reward': 1.}]) as run:
                    playback.main()
        self.assertFalse(run.call_args.args[3])
        self.assertEqual(run.call_args.args[1], 2)

    def test_video_capture_automatically_disables_live_display(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder)/'gameplay.mp4'
            with patch('sys.argv', ['play.py', '--video', str(video)]):
                with contextlib.redirect_stdout(io.StringIO()):
                    with patch.object(playback, 'play', return_value=[{'reward': 1.}]) as run:
                        playback.main()
            self.assertFalse(run.call_args.args[3])
            self.assertEqual(run.call_args.args[4], video)

    def test_bellman_targets_and_target_isolation(self):
        online = nn.Linear(1, 2, bias=False)
        target = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            online.weight.zero_()
            target.weight.copy_(torch.tensor([[10.], [4.]]))
        optimizer = torch.optim.SGD(online.parameters(), lr=.1)
        loss = train_step(online, target, optimizer, Batch().sample(2), .5)
        # Targets are 1 and 2 + .5*10 = 7. Huber losses .5 and 6.5.
        self.assertAlmostEqual(loss, 3.5)
        self.assertTrue(torch.equal(target.weight, torch.tensor([[10.], [4.]])))
        self.assertIsNone(target.weight.grad)
        self.assertTrue(torch.all(online.weight > 0))
        sync_target(online, target)
        self.assertTrue(torch.equal(online.weight, target.weight))
        self.assertFalse(target.training)
        self.assertFalse(target.weight.requires_grad)

    def test_nonfinite_update_is_rejected(self):
        online = nn.Linear(1, 2, bias=False)
        target = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            target.weight.fill_(float('nan'))
        before = online.weight.detach().clone()
        with self.assertRaises(FloatingPointError):
            train_step(online, target, torch.optim.SGD(online.parameters(), lr=.1),
                       Batch().sample(2), .5)
        self.assertTrue(torch.equal(before, online.weight))

    def test_log_schema_precision_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = SimpleNamespace(member=folder, experiment=1)
            logger = Logger(settings)
            logger.log_episode(1, 100, -21, 100, .12345678, .000000123)
            with open(logger.path) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(float(rows[0]['loss']), .000000123)
            self.assertEqual(tuple(rows[0]), Logger.fields)
            self.assertNotIn(b'\r', Path(logger.path).read_bytes())
            with self.assertRaises(FileExistsError):
                Logger(settings)

    def test_invalid_training_intervals_fail_before_environment_creation(self):
        for option in ('--train_every', '--target_update_every',
                       '--epsilon_decay_steps', '--save_every_episodes'):
            with self.subTest(option=option):
                with patch('sys.argv', ['train.py', '--member', 'Samuel',
                                        '--experiment', '1', option, '0']):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            get_settings()
                self.assertEqual(raised.exception.code, 2)

    def test_model_path_from_main_is_honored(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'custom/checkpoint.pth'
            settings = SimpleNamespace(member='Samuel', experiment=1, model_path=str(path))
            self.assertEqual(save_model(nn.Linear(1, 2), settings), str(path))
            self.assertTrue(path.exists())

    def test_single_episode_plotting_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'single.csv'
            log.write_text('episode,step,reward,length,epsilon,loss\n1,100,-21,100,0.5,0.01\n')
            output = Path(folder) / 'plots'
            script = Path(__file__).resolve().parents[1] / 'plot_results.py'
            subprocess.run([sys.executable, str(script), str(log), '--out_dir', str(output)],
                           env=dict(os.environ, MPLCONFIGDIR=str(Path(folder)/'mpl')),
                           check=True, capture_output=True)
            self.assertEqual(len(list(output.glob('*.png'))), 4)
            self.assertTrue(all(p.stat().st_size > 0 for p in output.glob('*.png')))

    def test_atomic_save_keeps_previous_file_on_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'nested/model.pth'
            model = nn.Linear(1, 2)
            settings = SimpleNamespace(member='Samuel', experiment=1)
            save_model(model, settings, path)
            original = path.read_bytes()
            with patch('train.torch.save', side_effect=OSError('simulated write failure')):
                with self.assertRaises(OSError):
                    save_model(model, settings, path)
            self.assertEqual(original, path.read_bytes())
            self.assertEqual(list(path.parent.glob('*.tmp')), [])
            state = torch.load(path, weights_only=True)
            self.assertTrue(torch.equal(state['weight'], model.weight))


if __name__ == '__main__':
    unittest.main()
