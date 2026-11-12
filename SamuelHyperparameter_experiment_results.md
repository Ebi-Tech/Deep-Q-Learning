# Member Name: Samuel

Budget: 20,000 environment decisions per run; replay capacity 2,000; warm-up 1,000; train every 4 decisions; target sync every 1,000. Training seed: 0.

Evaluation: 5 greedy episodes per model, seeds starting at 100. Raw rewards are reported. No capped episodes are treated as wins.

These single-seed pilot runs do not establish convergence or statistical superiority. Epsilon schedules longer than the budget do not reach their endpoint.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 100000 | baseline; 21 training episodes; last-10 raw return -20.30; evaluation mean -21.00, std 0.00; 0 evaluation episodes capped |
| 2 | 0.0002 | 0.99 | 32 | 1.0 | 0.01 | 100000 | higher learning rate; 21 training episodes; last-10 raw return -20.20; evaluation mean -21.00, std 0.00; 0 evaluation episodes capped |
| 3 | 2.5e-05 | 0.99 | 32 | 1.0 | 0.01 | 100000 | lower learning rate; 22 training episodes; last-10 raw return -20.30; evaluation mean -21.00, std 0.00; 0 evaluation episodes capped |
| 4 | 0.0001 | 0.97 | 32 | 1.0 | 0.01 | 100000 | lower discount; 22 training episodes; last-10 raw return -20.50; evaluation mean -21.00, std 0.00; 0 evaluation episodes capped |
| 5 | 0.0001 | 0.995 | 32 | 1.0 | 0.01 | 100000 | higher discount; 21 training episodes; last-10 raw return -19.80; evaluation mean -21.00, std 0.00; 0 evaluation episodes capped |
| 6 | 0.0001 | 0.99 | 16 | 1.0 | 0.01 | 100000 | smaller batch; 22 training episodes; last-10 raw return -20.50; evaluation mean -21.00, std 0.00; 0 evaluation episodes capped |
| 7 | 0.0001 | 0.99 | 64 | 1.0 | 0.01 | 100000 | larger batch; 22 training episodes; last-10 raw return -20.60; evaluation mean -20.40, std 0.49; 0 evaluation episodes capped |
| 8 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 100000 | higher final epsilon; 21 training episodes; last-10 raw return -20.30; evaluation mean -21.00, std 0.00; 0 evaluation episodes capped |
| 9 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 20000 | faster exploration decay; 23 training episodes; last-10 raw return -20.70; evaluation mean -21.00, std 0.00; 0 evaluation episodes capped |
| 10 | 0.0001 | 0.99 | 32 | 0.7 | 0.01 | 100000 | lower initial epsilon; 22 training episodes; last-10 raw return -20.30; evaluation mean -21.00, std 0.00; 0 evaluation episodes capped |

Machine-readable configurations and evaluation episodes are in `logs/Samuel/experiment_<n>.json`; raw training metrics are in the matching CSVs. Models are in `models/Samuel/`.

## Interpretation

Experiment 7 (batch size 64) had the highest measured greedy evaluation mean,
-20.4; all other experiments averaged -21. All models still lost their evaluation
episodes. This single-seed, 20,000-step comparison does not establish that batch
size 64 is superior or that training has converged. More training and repeated
training seeds are needed before selecting final hyperparameters.

The fixed budget also limits the exploration comparison: most schedules have
not reached their final epsilon by step 20,000. Training return uses exploratory
actions, while evaluation is greedy; these quantities should not be confused.

[Comparison plot](plots/Samuel/experiment_comparison.png),
[full measured summary](logs/Samuel/summary.json), and
[source/environment manifest](logs/Samuel/run_manifest.json).
