"""Regression checks for Samuel's training and persistence contracts."""
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from train import Logger, save_model, sync_target, train_step


class Batch:
    def __len__(self):
        return 2

    def sample(self, batch_size):
        assert batch_size == 2
        # A terminal transition must ignore the target value of 10.
        return (torch.ones(2, 1), torch.tensor([0, 1]), torch.tensor([1., 2.]),
                torch.ones(2, 1), torch.tensor([1., 0.]))


class SamuelTests(unittest.TestCase):
    def test_bellman_targets_and_target_isolation(self):
        online = nn.Linear(1, 2, bias=False)
        target = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            online.weight.zero_()
            target.weight.copy_(torch.tensor([[10.], [4.]]))
        optimizer = torch.optim.SGD(online.parameters(), lr=.1)
        loss = train_step(online, target, optimizer, Batch(),
                          SimpleNamespace(batch_size=2, gamma=.5))
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
                       Batch(), SimpleNamespace(batch_size=2, gamma=.5))
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
