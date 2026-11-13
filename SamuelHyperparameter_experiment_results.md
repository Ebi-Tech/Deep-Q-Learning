# Member Name: Samuel

Budget: 20,000 environment decisions per run; replay capacity 2,000; warm-up 1,000; train every 4 decisions; target sync every 1,000. Training seed: 0.

Evaluation: 5 greedy episodes per model, seeds starting at 100. Raw rewards are reported. No capped episodes are treated as wins.

These single-seed pilot runs do not establish convergence or statistical superiority. Epsilon schedules longer than the budget do not reach their endpoint.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 100000 | baseline; 20 training episodes; last-10 raw return -20.20; evaluation mean -20.00, std 0.63; 0 evaluation episodes capped |
| 2 | 0.0002 | 0.99 | 32 | 1.0 | 0.01 | 100000 | higher learning rate; 21 training episodes; last-10 raw return -20.00; evaluation mean -21.00, std 0.00; 0 evaluation episodes capped |
| 3 | 2.5e-05 | 0.99 | 32 | 1.0 | 0.01 | 100000 | lower learning rate; 22 training episodes; last-10 raw return -20.10; evaluation mean -20.80, std 0.40; 0 evaluation episodes capped |
| 4 | 0.0001 | 0.97 | 32 | 1.0 | 0.01 | 100000 | lower discount; 21 training episodes; last-10 raw return -20.70; evaluation mean -21.00, std 0.00; 0 evaluation episodes capped |
| 5 | 0.0001 | 0.995 | 32 | 1.0 | 0.01 | 100000 | higher discount; 21 training episodes; last-10 raw return -20.40; evaluation mean -21.00, std 0.00; 0 evaluation episodes capped |
| 6 | 0.0001 | 0.99 | 16 | 1.0 | 0.01 | 100000 | smaller batch; 21 training episodes; last-10 raw return -20.30; evaluation mean -21.00, std 0.00; 0 evaluation episodes capped |
| 7 | 0.0001 | 0.99 | 64 | 1.0 | 0.01 | 100000 | larger batch; 22 training episodes; last-10 raw return -20.50; evaluation mean -21.00, std 0.00; 0 evaluation episodes capped |
| 8 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 100000 | higher final epsilon; 21 training episodes; last-10 raw return -20.20; evaluation mean -20.80, std 0.40; 0 evaluation episodes capped |
| 9 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 20000 | faster exploration decay; 22 training episodes; last-10 raw return -20.80; evaluation mean -21.00, std 0.00; 0 evaluation episodes capped |
| 10 | 0.0001 | 0.99 | 32 | 0.7 | 0.01 | 100000 | lower initial epsilon; 21 training episodes; last-10 raw return -19.80; evaluation mean -21.00, std 0.00; 0 evaluation episodes capped |

Machine-readable configurations and evaluation episodes are in `logs/Samuel/main_20261113/experiment_<n>.json`; raw training metrics are in the matching CSVs. Models: `models/Samuel/main_20261113/`. Run ID: main_20261113.

## Integration rerun — 13 November 2026

Merged `origin/main` at `f6e310a` into `samuel`; the training source for this
rerun is `a60a968`. This includes the colleagues' Kaiming initialization,
uint8 replay sampling, adjusted Adam epsilon, periodic model saving, and the
single-episode plotting fix. Samuel's training function now accepts a sampled
batch and gamma, matching the new main-loop call. Atomic saving honors
`--model_path`.

All ten runs completed (200,000 total training decisions). All 50 evaluation
episodes finished without a step cap. All saved checkpoints reload, contain
finite parameters, and produce finite inference outputs. All logged losses
are finite. Seven regression cases passed, and a separate real Pong smoke run
tested main's default training interval of 1, periodic saving, and a custom
checkpoint path. The former single-episode plotting failure also passed.

The explicitly selected interval of 4 preserves the previous pilot's cadence;
normal training still defaults to interval 1. Other shared budgets are shown
above. This is a controlled short pilot, not a full million-step run.

Experiment 1 had the highest observed greedy mean, -20.00. The original
pilot selected experiment 7 at -20.4. Multiple implementation details changed
and only one training seed was used; this comparison does not isolate a causal
effect or establish statistically reliable superiority. Negative episode returns
indicate weak gameplay, not a Python exception. More training is needed.

[New comparison plot](plots/Samuel/main_20261113/experiment_comparison.png),
[full summary](logs/Samuel/main_20261113/summary.json),
[source/environment manifest](logs/Samuel/main_20261113/run_manifest.json), and
[gameplay](videos/Samuel/main_20261113/gameplay.mp4).

[Original pilot report](logs/Samuel/pilot_20261112_results.md) and its raw CSVs,
models, plots, and video were preserved rather than overwritten.

## Best, worst, and interpretation of the ten pilots

All comparisons below use the same 20,000-decision budget, seed and five-episode
greedy test. **Experiment 1 was the pilot best at -20.0. Experiments 2, 4, 5,
6, 7, 9 and 10 tied for worst at -21.0.** Experiments 3 and 8 averaged -20.8.
Every pilot still lost its test games; small differences do not establish a
reliable long-run ranking.

| Hyperparameter investigated | Observed result | What I can conclude |
|---|---|---|
| Learning rate: experiments 2 and 3 | 0.0002 gave -21.0; 0.000025 gave -20.8; baseline 0.0001 gave -20.0 | Neither alternative improved this short pilot. Longer runs are needed to distinguish update speed from instability. |
| Gamma: experiments 4 and 5 | 0.97 and 0.995 both gave -21.0 | No improvement was observed; these pilots do not establish which discount is best after learning. |
| Batch size: experiments 6 and 7 | 16 and 64 both gave -21.0; baseline 32 gave -20.0 | Larger/smaller batches did not help at this budget; I retained 32 for the full baseline. |
| Epsilon end: experiment 8 | 0.05 gave -20.8 versus -20.0 for baseline 0.01 | The 100,000-step schedules have not finished at 20,000 steps, so this is not a clean test of the final exploration floor. |
| Epsilon decay: experiment 9 | Reaching 0.01 in 20,000 steps gave -21.0 | Faster decay did not help this run. Exploiting an immature policy is a plausible explanation, not a demonstrated causal effect. |
| Epsilon start: experiment 10 | Starting at 0.7 gave -21.0 versus -20.0 at 1.0 | Lower initial exploration did not help this pilot; one seed is insufficient to generalize. |

I retained experiment 1's configuration for the full baseline: learning rate
0.0001, gamma 0.99, batch size 32, epsilon 1.0 to 0.01 over 100,000 steps. It had
the highest measured pilot return, but the pilot gap was small. The full run
then supplied enough training to assess whether that baseline could learn.

The full run additionally changed the budget to one million, training interval
from 4 to 1, replay capacity from 2,000 to 50,000, warm-up from 1,000 to 10,000,
and execution device from CPU to MPS. Its improvement cannot be attributed to
one table hyperparameter. The separate 20-game test below supports the chosen
trained checkpoint; it does not show that this is the globally optimal setting.

## Completed full-length baseline — experiment 11

The separate run `full_20261113` completed **1,000,000 decisions**, training every step
on Apple MPS. Baseline hyperparameters: learning rate 0.0001, gamma 0.99, batch
size 32, epsilon 1.0 → 0.01 over 100,000 steps. Replay capacity was 50,000;
warm-up 10,000; target synchronization every 1,000 decisions; training seed 0.

| Result | Measured value |
|---|---:|
| Total training decisions | 1,000,000 |
| Selected checkpoint step | 974,891 |
| Last 100 training episodes: mean return | +6.39 |
| Checkpoint selection mean, 5 episodes | +13.80 |
| Independent mean return, 20 episodes | **+11.25** |
| Independent standard deviation | 4.62 |
| Independent wins / losses | **19 / 1** |
| Truncated or capped independent episodes | 0 |

Checkpoint selection used seeds 100–104. Independent testing used seeds
1000–1019. The strongest checkpoint was retained at step 974,891;
the actual million-step final weights are also saved. This single full baseline
is separate from the ten short pilot experiments above. It is not ten full-length
experiments. Its stronger performance does not guarantee future wins.

[Full measured report](logs/Samuel/full_20261113/report.md),
[independent evaluation](logs/Samuel/full_20261113/final_evaluation.json),
[learning curve](plots/Samuel/full_20261113/full_training_evaluation.png), and
[winning gameplay](videos/Samuel/full_20261113/gameplay.mp4).
