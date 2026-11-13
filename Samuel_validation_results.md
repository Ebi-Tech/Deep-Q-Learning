# Samuel: DQN integration validation

Validated on 10 November 2026 on branch `samuel`, against source commit
`e6f77f8`. These are smoke tests of the existing implementation, not the
ten hyperparameter experiments or evidence of a trained Pong policy.
No teammate source code was changed.

## Environment

- macOS ARM64, Python 3.11.18, CPU execution.
- Installed the dependencies from `uv.lock` with `uv sync --locked`.
- `uv pip check` confirmed all 26 installed packages were compatible.
- All project Python files passed syntax parsing.
- Pong reset returned uint8 observations with shape `(4, 84, 84)`.
- The environment exposed six actions; RGB rendering returned `(210, 160, 3)`.

## Samuel's implemented training interfaces

- `train_step`: returned finite loss and updated online network weights.
  Target weights stayed unchanged and target parameters received no gradients.
- `sync_target`: copied online weights into the target network exactly.
- `Logger.log_episode`: wrote the expected CSV fields and episode row.
- `save_model`: saved a state dictionary that loaded into a fresh `QNetwork`
  with identical weights.

Replay sampling, CNN batch output, environment stepping, and both random and
greedy action selection also passed runtime checks.

## Short training run

Executed the actual training CLI with:

```bash
python train.py --member Samuel_smoke --experiment 2 --total_steps 4500 \
  --buffer_size 128 --learning_starts 32 --batch_size 8 \
  --train_every 16 --target_update_every 64
```

Other hyperparameters used the defaults. The small buffer and short budget
were chosen to validate integration, not measure learning performance.

| Episode | Cumulative steps | Raw reward | Episode length | Latest loss |
|---|---:|---:|---:|---:|
| 1 | 1009 | -20 | 1009 | 0.000008 |
| 2 | 1792 | -21 | 783 | 0.000063 |
| 3 | 2556 | -21 | 764 | 0.000132 |
| 4 | 3462 | -21 | 906 | 0.000097 |
| 5 | 4467 | -20 | 1005 | 0.000032 |

Training completed and saved its final model. All four plot outputs were
successfully generated from this five-episode CSV. The negative rewards do
not demonstrate a successful policy.

## Outstanding work and observed issue

- `plot_results.py` fails on a one-episode CSV with
  `TypeError: len() of unsized object`. Multi-episode plotting passed.
- `play.py` contains only a comment; playback remains unimplemented.
- Samuel's architecture diagram, gameplay video, and ten hyperparameter
  experiments remain outstanding.
- The PowerShell experiment runner was not executed because PowerShell was
  unavailable. Full-budget training was not performed.
- No rubric was available, so rubric compliance was not assessed.

Temporary logs, models, and plots were produced outside the repository at
`/private/tmp/samuel-dqn-check-rssxe52i`. They are machine-local test artifacts,
not committed deliverables, and may be removed by temporary-file cleanup.

## Implementation follow-up — 12 November 2026

The outstanding list above describes the original 10 November review. Samuel's
implementation is now committed in `04328ef`:

- `train_step` validates replay readiness, rejects non-finite loss/gradients,
  retains vanilla DQN Bellman targets and Huber loss, and updates only the online network.
- `sync_target` also freezes target parameters and enforces evaluation mode.
- CSV logging retains full numeric precision and refuses to overwrite prior runs.
- Checkpoint saving creates destination directories and atomically replaces weights;
  a simulated failed write verified that the previous checkpoint survives.
- `play.py` loads compatible weights, evaluates seeded episodes, reports step caps,
  and records actual gameplay to MP4. No generated or simulated gameplay footage is used.
- Four regression tests passed: exact terminal/nonterminal Bellman targets and
  target isolation; non-finite update rejection; CSV schema/precision and overwrite
  protection; atomic-save failure recovery and checkpoint reload.
- Divine's settings/environment/CNN and Gabriel's replay/main-loop code were verified
  unchanged. Gabriel's one-episode plotting issue remains unchanged.

All ten pilot experiments completed 20,000 decisions each (200,000 total), with
five greedy evaluation episodes each (50 total). All ten CSVs, evaluation records,
and saved model state dictionaries were checked. Model parameters and logged
losses were finite, and every evaluation episode completed without a step cap.
Forty per-run plots and a comparison plot were generated.

Experiment 7 had the highest measured evaluation mean (-20.4). All other models
averaged -21. This is not evidence of a strong policy, convergence, or statistically
reliable hyperparameter superiority. Only one training seed and a short budget
were used; full-budget training and rubric compliance remain unverified.

The selected model's recorded episode (seed 100) returned -21. Its MP4 was fully
decoded and verified: 817 frames at 15 fps, approximately 54.47 seconds. See
`videos/Samuel/gameplay.mp4` and its JSON episode metadata. Architecture diagrams
are available as PNG, SVG, and Mermaid in `docs/`. The completed experiment table,
raw metrics, configuration/source manifest, model files, and plots are stored in
the repository rather than temporary directories.

## Merge and rerun — 13 November 2026

Merged all fetched `origin/main` changes through `f6e310a` (merge `2747b8c`).
Resolved the training-call conflict by using the new sampled-batch/gamma
interface, and reconciled atomic saving with main's `--model_path` and periodic
save calls. Source commit for the rerun: `a60a968`.

The main environment, CNN, replay, exploration and main-loop definitions were
checked against upstream and retain its implementation exactly. Added argument
validation rejects zero scheduling intervals before creating the environment.
Seven regression cases passed, including model-path handling, single-episode
plotting and invalid scheduling arguments. A real 1,100-step Pong integration
run exercised training every step, saving every episode, a custom model path,
and plotting its finished episode.

All ten 20,000-step reruns completed under run ID `main_20261113`, with train interval
4 explicitly retained for comparison; main's default remains 1. All 50 greedy
evaluations finished without step caps. Every new model reloads and has finite
parameters and inference outputs; all logged losses are finite. Generated
40 per-run plots plus the comparison figure. The selected experiment is 1
with observed mean -20.00; the new recorded episode scored -21
and was decoded fully (846 frames, 15 fps, 56.40 seconds).

All original Samuel CSVs were verified byte-for-byte against the prior commit,
and all original model hashes still match their saved summary. The previous
results report is archived at `logs/Samuel/pilot_20261112_results.md`.

The remaining limitation is training quality and budget, not a failure observed
in these runs. These pilots do not establish convergence or rubric compliance.

## Full baseline completed

Run `full_20261113` reached exactly 1,000,000 decisions on Apple MPS, training every
step. The job completed all checkpoint evaluations, a 20-episode independent
test, gameplay recording, and plotting. Total wall time was
2.51 hours. There were 383 completed training episodes;
a partial last episode is included in the step budget but not the episode CSV.

The selected checkpoint at step 974,891 had a five-episode selection
mean of +13.80. On separate seeds 1000–1019 it averaged
+11.25 (standard deviation 4.62), with
19 wins and 1 loss. All 20 games terminated normally; none were
truncated or capped. The final training mean over 100 episodes was +6.39.

Selected, latest and actual million-step checkpoint weights reload and produce
finite inference outputs. All logged training losses are finite. The selected
model matches its numbered snapshot; latest matches final weights byte-for-byte.
The full winning gameplay clip was decoded: 3,348 frames, 15 fps, 223.2 seconds,
seed 1000, raw return +4. The ten pilot experiments remain separate. This is one
training seed and a finite evaluation sample, not a guarantee of future wins.
