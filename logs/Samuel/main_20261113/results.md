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
