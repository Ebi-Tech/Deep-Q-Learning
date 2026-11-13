# Samuel: full-length baseline

The completed `full_20261113` run reached one million decisions. The selected
model averaged **+11.25 over 20 independent games**, winning **19 of 20**.
[Measured results](../logs/Samuel/full_20261113/report.md).

`run_long_samuel.py` runs a baseline using the shared DQN components for one
million environment decisions and a gradient update every step. This is separate from the ten
20,000-step pilots. The launcher chooses CUDA, Apple MPS, or CPU, without
changing teammates' training code. The current utility calls shared components
from a plain training loop with separate status/evaluation functions; it does
not replace shared logger or saving functions at runtime.

```bash
uv run run_long_samuel.py --run_id full_run_02 --steps 1000000 --device auto
```

Choose a fresh run ID. The baseline keeps the shared defaults: learning rate
0.0001, gamma 0.99, batch size 32, replay capacity 50,000, warm-up 10,000,
epsilon 1.0 to 0.01 over 100,000 steps, and target synchronization every
1,000 decisions. Checkpoints are saved every 25 completed episodes.

For run ID `full_20261113`, the automatically updated local outputs are:

| Output | Location |
|---|---|
| Phase, step count, training return, loss and estimated time | `logs/Samuel/full_20261113/status.json` |
| Readable progress and final result | `logs/Samuel/full_20261113/report.md` |
| Raw episode metrics | `logs/Samuel/full_20261113/experiment_11.csv` |
| Exact settings and source hashes | `logs/Samuel/full_20261113/manifest.json` |
| Checkpoint selection evaluations | `logs/Samuel/full_20261113/checkpoint_evaluations.json` |
| Final independent evaluation | `logs/Samuel/full_20261113/final_evaluation.json` |
| Latest, best and numbered model snapshots | `models/Samuel/full_20261113/` |
| Final plots | `plots/Samuel/full_20261113/` |
| Final recorded gameplay | `videos/Samuel/full_20261113/` |

The worker selects the checkpoint with the highest observed mean over five
greedy episodes, using seeds 100–104. It excludes candidates with capped
evaluations. At the end, it tests the selected model on 20 separate-seed
episodes (1000–1019), then records gameplay and generates plots. Selection
scores and independent evaluation scores are reported separately.

The phase becomes `completed` only after those final steps succeed. On a Python
failure, `status.json` reports `failed` and `error.txt` contains the traceback.
Snapshots contain inference weights, not the replay buffer and optimizer state
needed for exact training resumption.

The completed local run used `caffeinate -i` to prevent idle sleep until the job
exited. The computer must remain powered on for local training to continue.
More steps give the agent more opportunity to learn; strong performance must
be established by evaluation, not assumed from the training budget. No full-run
scores should be entered in the pilot table before evaluation completes.

Regenerate the full learning/evaluation figure with:

```bash
uv run plot_full_training.py --run_id full_20261113
```

## Training provenance and submission playback

The completed million-step artifacts were generated with the earlier launcher
at commit `860ba86`, identified by their original source hashes. Later launcher
readability changes and playback CLI corrections do not change those measured
results. The original implementation remains available in Git history.

The root `dqn_model.pth` is a byte-identical export of the selected `best.pth`.
Run `uv run play.py` for the default greedy live demonstration, or pass
`--headless` for automated evaluation. Use the recorded video in the README;
use the live script for the presentation.
