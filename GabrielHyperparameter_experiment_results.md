# Member Name: Gabriel

All runs: 1,000,000 steps, `--train_every 1` (with the original value of 4 the baseline did not learn). Focus: epsilon decay and epsilon end. Each run uses one seed, so differences between runs can be partly luck.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 100,000 | Learned. Full 1,000,000 steps (998,869 logged). Reward stayed near -20 until about step 60k, rose above -15 by step 161k, above 0 by step 341k and averaged +8.1 over the last 10 episodes (+8.3 over the last 50; peak 10-episode average +11.9; best game +18). Episode length grew from about 800 to about 2,850 steps as the agent began returning the ball. Results vary between runs of this same setting: an earlier run also learned (about +10) but a third run stayed near -21 and was stopped at 684k steps. Single seed. |
| 2 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 25,000 | Learned despite the very fast decay: epsilon reached 0.01 by step 25.8k. Reward averaged -17.1 over the 10 episodes before step 100k, rose above -15 by step 118k and above 0 by step 527k, peaked at a 10-episode average of +9.8 (best game +17) and ended at +5.7 over the last 10 episodes (+6.6 over the last 50). Episode length grew to about 3,040 steps. Early progress was faster than the baseline's decay of 100k at first (about -17 at step 100k) but the final score was a little lower (+5.7 vs +8.1 for experiment 1); with one seed per run this gap may be luck. 1,000,000 steps (999,387 logged). |
| 3 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 50,000 | |
| 4 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 200,000 | |
| 5 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 300,000 | |
| 6 | 1e-4 | 0.99 | 32 | 1.0 | 0.001 | 100,000 | |
| 7 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 100,000 | |
| 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.1 | 100,000 | |
| 9 | 1e-4 | 0.99 | 32 | 1.0 | 0.1 | 300,000 | |
| 10 | 1e-4 | 0.99 | 32 | 1.0 | best (TBD) | best (TBD) | |
