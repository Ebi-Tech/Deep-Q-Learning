# Member Name: Samuel

## Notebook study: batch size and starting epsilon (completed 15 November 2026)

Ten full runs from [Samuel_DQN_experiments.ipynb](notebooks/Samuel_DQN_experiments.ipynb), run ID
`notebook_batch_epsilon_01`, source commit `b09aa8f`. Each run: 1,000,000 training steps, replay buffer
50,000, warm-up 10,000, train every step, target sync every 1,000, training seed 0. "Won a game" means an
episode reward above 0. "Test" means 5 greedy games on seeds 100-104; none was capped or truncated.
Experiments 6 and 9 were trained on a Kaggle Tesla T4 GPU with the same commit, settings and seed, then
evaluated on this Mac like the others; all other runs were trained on Apple MPS.

| Exp | Learning rate | Gamma | Batch | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 100000 | Baseline. First won a game at ~231k steps; its 10-game average became positive at ~417k and passed +10 at ~722k. Fairly stable at the end: won 82 of the last 100 games (scores -11 to +21). **Last-100 average +6.39**; test +10.2 (5/5 wins). Full exploration from 1.0 fills the early replay buffer with random play, so learning starts later, but the standard batch gives steady updates. |
| 2 | 0.0001 | 0.99 | **8** | 1.0 | 0.01 | 100000 | Slowest to take off: first win at ~358k, average positive only at ~731k, above +10 at ~865k. Won 87 of the last 100 (-14 to +17). **Last-100 average +6.73**; test +8.0 (4/5 wins). Very small batches give noisy gradient estimates, so learning started late, but once it caught on it improved quickly in the final 250k steps. |
| 3 | 0.0001 | 0.99 | **16** | 1.0 | 0.01 | 100000 | First win at ~284k, average positive at ~648k, never stayed above +10. Won 65 of the last 100 (-14 to +15). **Last-100 average +1.87**; test +10.0 (5/5 wins). Still-noisy updates slowed learning, and it was still improving when the budget ran out. |
| 4 | 0.0001 | 0.99 | **64** | 1.0 | 0.01 | 100000 | First win at ~455k, average positive at ~648k, never stayed above +10. Won 70 of the last 100 (-17 to +16). **Last-100 average +3.91**; test +9.8 (5/5 wins). With the same learning rate, larger batches average out the gradient and make smaller effective progress per step, so it learned more slowly than the baseline. |
| 5 | 0.0001 | 0.99 | **128** | 1.0 | 0.01 | 100000 | Slow and late: first win at ~331k, average positive only at ~717k, never stayed above +10. Won 69 of the last 100 (-13 to +16). **Last-100 average +3.28**; test +2.6 (4/5 wins, one game lost -7). The largest batch at the base learning rate makes the smoothest but smallest effective updates, so it was still catching up when the budget ended. |
| 6 | **0.0002** | 0.99 | **64** | 1.0 | 0.01 | 100000 | Learned quickly: first win at ~287k, average positive at ~338k, above +10 at ~647k. Very consistent: won 94 of the last 100 (-7 to +17). **Last-100 average +9.08**; test +11.2 (5/5 wins). Trained on a Kaggle T4 GPU. Doubling the learning rate compensated for the larger batch's smoother, smaller steps, so it far outperformed batch 64 at the base rate (exp 4). |
| 7 | **0.0004** | 0.99 | **128** | 1.0 | 0.01 | 100000 | First win at ~418k, average positive at ~687k, never stayed above +10 but was still rising at the end. Won 70 of the last 100 (-10 to +17). **Last-100 average +4.40**; test +10.2 (5/5 wins). Quadrupling the learning rate helped the largest batch (vs +3.28 for exp 5 at the base rate), but each 128-sample update still moves slowly, so it learned later than the smaller-batch runs. |
| 8 | 0.0001 | 0.99 | 32 | **0.5** | 0.01 | 100000 | Best run. First win at ~236k, average positive at ~310k, above +10 already at ~394k. Most stable: won 99 of the last 100 (-6 to +21). **Last-100 average +14.60**; test +15.4 (5/5 wins, std 1.7). Starting with less random play meant more early experience came from the agent's own policy, so useful rallies reached the replay buffer sooner. |
| 9 | 0.0001 | 0.99 | 32 | **0.75** | 0.01 | 100000 | First win at ~234k, average positive at ~320k, above +10 at ~428k. Won 98 of the last 100 (-13 to +20). **Last-100 average +12.92**; test +14.8 (5/5 wins). Trained on a Kaggle T4 GPU. Same effect as exp 8 but slightly weaker, consistent with more random play at the start. |
| 10 | 0.0001 | 0.99 | **8** | **0.5** | 0.01 | 100000 | Weakest of the full runs: first won a game only at ~582k steps and its 10-game average never became positive. Won just 15 of the last 100 games (-15 to +5), though still improving at the end. **Last-100 average -4.02**; test +0.2 (3/5 wins, scores -6 to +5). The two individually best settings did not combine well: batch 8's noisy, slow-starting updates (exp 2 only turned positive at ~731k) dominated, and the head start from epsilon 0.5 could not make up for them within one million steps. Chosen by the rule: best batch size by last-100 average among exps 1-5 is 8 (exp 2, +6.73; exp 1 +6.39, exp 4 +3.91, exp 5 +3.28, exp 3 +1.87); best epsilon start among exps 1, 8, 9 is 0.5 (exp 8, +14.60). Not a repeat of the baseline. |

Experiment 10's settings were chosen by the instructor's rule (highest last-100 training average for batch
size among experiments 1-5 and for starting epsilon among experiments 1, 8 and 9). It was started once
experiment 8 clearly led and was confirmed after experiments 5 and 9 finished. Single training seed per
configuration, so differences of a few points may be noise.

### Best model from this study: experiment 8

Chosen by the highest five-game selection score (+15.4). Independent test on 20 new seeds (1000-1019),
which did not influence the choice: **mean +15.25**, std 2.70, **20/20 wins** (scores +10 to +20), no
capped or truncated games. Weights: `Samuel_outputs_notebook_batch_epsilon_01/dqn_model.pth`; plots in
`plots/`; one recorded game (+13) in `gameplay.mp4`.

These weights are also the submitted root `dqn_model.pth`, which `uv run play.py` loads by default.

Sources: `Samuel_outputs_notebook_batch_epsilon_01/experiment_<n>/record.json`, `summary.json`,
`experiment_10_selection.json`, `independent_evaluation.json`; training CSVs in
`logs/Samuel/notebook_batch_epsilon_01/`; models in `models/Samuel/notebook_batch_epsilon_01/`.
