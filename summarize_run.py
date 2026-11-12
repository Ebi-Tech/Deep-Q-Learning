# summarize_run.py
# Prints the numbers you need for the "Observed behaviour" column.
# Usage: uv run summarize_run.py logs/Gabriel/experiment_3.csv

import sys

import numpy as np

data = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
reward, step, length = np.atleast_1d(data["reward"]), np.atleast_1d(data["step"]), np.atleast_1d(data["length"])
n = len(reward)
k = min(10, n)
print(f"Episodes: {n} | total steps: {int(step[-1])}")
print(f"Average reward, first {k} episodes: {reward[:k].mean():.1f}")
print(f"Average reward, last {k} episodes:  {reward[-k:].mean():.1f}")
print(f"Best episode reward: {reward.max():.0f} (episode {int(np.argmax(reward)) + 1}, step {int(step[np.argmax(reward)])})")
print(f"Average episode length, last {k}: {length[-k:].mean():.0f}")
above = np.where(np.convolve(reward, np.ones(k) / k, mode="valid") > -15)[0]
if len(above):
    print(f"10-episode average first rose above -15 at step {int(step[above[0] + k - 1])}")
else:
    print("10-episode average never rose above -15")
