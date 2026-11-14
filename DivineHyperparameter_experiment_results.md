# Member Name: Divine

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 100000 | Reward stayed near -20 while epsilon was high, then rose quickly after exploration ended. It became positive around 280k steps and levelled off at about +12 to +14 from 400k steps. Last 100 episodes average: +12.87. This is the reference for my other runs. |
| 2 | 0.00005 | 0.99 | 32 | 1.0 | 0.01 | 100000 | Learned the same way as the baseline but slower. It became positive around 420k steps instead of 280k, and was still improving at the end (last 10 episodes +16.2). Last 100 episodes average: +11.79, close to the baseline. A lower learning rate made learning slower but still stable. |
| 3 | 0.00025 | 0.99 | 32 | 1.0 | 0.01 | 100000 | Learned faster at first and became positive around 290k steps, but never settled. Rewards kept swinging between about -10 and +18 right up to the end. Last 100 episodes average: +1.35. The higher learning rate made updates too big, so the agent kept overshooting and losing what it had learned. |
| 4 | 0.0005 | 0.99 | 32 | 1.0 | 0.01 | 100000 | Same unstable pattern as Exp 3, slightly better on average. It reached +19 in single episodes but also dropped to -11 in the same period. Last 100 episodes average: +5.52. Still far below the baseline, and too noisy to be reliable. |
| 5 | 0.001 | 0.99 | 32 | 1.0 | 0.01 | 100000 | Worst run. Training loss was 5 to 10 times higher than the other runs, and the agent stayed between about -15 and -5 for most of training. It won its first game only at 725k steps. Last 100 episodes average: -8.04. The learning rate was too high, so training was unstable from the start. |
| 6 | 0.0001 | 0.95 | 32 | 1.0 | 0.01 | 100000 | Learned, but slower than the baseline. It became positive around 500k steps and only reached +5 to +13 in the last 200k. Last 100 episodes average: +2.77. With gamma 0.95 the agent only values rewards about 20 steps ahead, which is shorter than a rally, so linking a hit to a point was harder. |
| 7 | | | | | | | |
| 8 | | | | | | | |
| 9 | | | | | | | |
| 10 | | | | | | | |

Every run trained for 1,000,000 steps with the same fixed settings, and only the setting named in its row was changed from the baseline. Experiments are compared by the average reward over the last 100 training episodes.
