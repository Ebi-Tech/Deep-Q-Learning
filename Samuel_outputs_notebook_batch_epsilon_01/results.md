# Samuel: notebook experiment results

| Exp | Learning rate | Gamma | Batch | Epsilon start | Epsilon end | Decay |
|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 100000 |
| 2 | 0.0001 | 0.99 | 8 | 1.0 | 0.01 | 100000 |
| 3 | 0.0001 | 0.99 | 16 | 1.0 | 0.01 | 100000 |
| 4 | 0.0001 | 0.99 | 64 | 1.0 | 0.01 | 100000 |
| 5 | 0.0001 | 0.99 | 128 | 1.0 | 0.01 | 100000 |
| 6 | 0.0002 | 0.99 | 64 | 1.0 | 0.01 | 100000 |
| 7 | 0.0004 | 0.99 | 128 | 1.0 | 0.01 | 100000 |
| 8 | 0.0001 | 0.99 | 32 | 0.5 | 0.01 | 100000 |
| 9 | 0.0001 | 0.99 | 32 | 0.75 | 0.01 | 100000 |
| 10 | 0.0001 | 0.99 | 8 | 0.5 | 0.01 | 100000 |

| Exp | Steps | Last-100 training mean | Greedy selection mean (5 games) |
|---|---|---|---|
| 1 | 1,000,000 | +6.39 | +10.20 |
| 2 | 1,000,000 | +6.73 | +8.00 |
| 3 | 1,000,000 | +1.87 | +10.00 |
| 4 | 1,000,000 | +3.91 | +9.80 |
| 5 | 1,000,000 | +3.28 | +2.60 |
| 6 | 1,000,000 | +9.08 | +11.20 |
| 7 | 1,000,000 | +4.40 | +10.20 |
| 8 | 1,000,000 | +14.60 | +15.40 |
| 9 | 1,000,000 | +12.92 | +14.80 |
| 10 | 1,000,000 | -4.02 | +0.20 |

Selection scores use seeds 100–104; they are distinct from training returns and the final independent test. One training seed per configuration.

## Selected model: experiment 8

Independent mean return: **+15.25**; standard deviation 2.70; wins **20/20**. Seeds 1000–1019.
