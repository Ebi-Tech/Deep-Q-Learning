# Samuel: completed one-million-step baseline

**Completed: 1,000,000 environment decisions, training every step on Apple MPS.**
The selected checkpoint achieved a mean raw return of **+11.25**
on 20 independent evaluation episodes and won **19 of 20**.

| Measurement | Result |
|---|---:|
| Training decisions | 1,000,000 |
| Completed training episodes | 383 |
| Last 100 training episodes: mean raw return | +6.39 |
| Selected checkpoint step | 974,891 |
| Checkpoint selection mean (5 episodes, seeds 100–104) | +13.80 |
| Independent test mean (20 episodes, seeds 1000–1019) | +11.25 |
| Independent test standard deviation | 4.62 |
| Independent wins / losses | 19 / 1 |
| Truncated or capped independent episodes | 0 |
| Wall time, including evaluation and recording | 2.51 hours |

The whole run reached one million steps. The checkpoint at step
974,891 was retained because it had the highest checkpoint-selection
mean; `best.pth` is that model. `final_001000000.pth` and `latest.pth` contain
the actual end-of-training weights. The final test evaluates the selected model,
not an average of different checkpoints.

The baseline used learning rate 0.0001, gamma 0.99, batch size 32, replay capacity
50,000, warm-up 10,000, target sync every 1,000 decisions, and Adam epsilon
0.00015. Exploration decayed from 1.0 to 0.01 over 100,000 steps. Training seed
was 0. This is one training run; the 20 test episodes are not 20 training seeds.

The earlier 20,000-step pilots had a highest observed mean of -20.0 after the
main merge. The longer baseline now demonstrates substantially stronger Pong
play on its independent test. These evaluations use different seeds and budgets;
they do not isolate a causal effect of any one setting or guarantee future wins.
One of the 20 independent games was lost.

## Verified artifacts

- [Raw training CSV](experiment_11.csv): finite losses; partial final episodes are not logged.
- [Checkpoint selection episodes](checkpoint_evaluations.json).
- [Independent test episodes](final_evaluation.json).
- [Settings and source hashes](manifest.json).
- [Verified model hashes and outcome summary](verified_summary.json).
- [Selected model](../../../models/Samuel/full_20261113/best.pth).
- [Actual million-step model](../../../models/Samuel/full_20261113/final_001000000.pth).
- [Learning curve and independent outcomes](../../../plots/Samuel/full_20261113/full_training_evaluation.png).
- [Winning gameplay recording](../../../videos/Samuel/full_20261113/gameplay.mp4).

The recorded episode uses seed 1000 and returns +4. Its full 3348-frame
video was decoded and checked: 15 fps, 223.2 seconds. The selected,
latest and final checkpoints reload with finite parameters and inference outputs.
The original ten pilot reports, logs and models remain separate and unchanged.
