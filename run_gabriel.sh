#!/usr/bin/env bash
# Gabriel's experiments: epsilon decay and epsilon end (1,000,000 steps each).
# Usage:  bash run_gabriel.sh 1 2 3     (runs experiments 1, 2 and 3 one after another)
# Experiment 10 is not here: it uses the best decay + best end, decided after 1-9.
#   Example:  uv run train.py --member Gabriel --experiment 10 --total_steps 1000000 \
#                 --epsilon_decay_steps <best> --epsilon_end <best>
# Baseline for every run: lr 1e-4, gamma 0.99, batch 32, eps start 1.0, eps end 0.01, decay 100000
STEPS=1000000

run() {
  n=$1; shift
  echo "=== Experiment $n: $* ==="
  uv run train.py --member Gabriel --experiment "$n" --total_steps $STEPS "$@"
}

for n in "$@"; do
  case $n in
    1) run 1 ;;                                                            # baseline
    2) run 2 --epsilon_decay_steps 25000 ;;
    3) run 3 --epsilon_decay_steps 50000 ;;
    4) run 4 --epsilon_decay_steps 200000 ;;
    5) run 5 --epsilon_decay_steps 300000 ;;
    6) run 6 --epsilon_end 0.001 ;;
    7) run 7 --epsilon_end 0.05 ;;
    8) run 8 --epsilon_end 0.1 ;;
    9) run 9 --epsilon_decay_steps 300000 --epsilon_end 0.1 ;;             # heavy exploration
    *) echo "Experiment $n is not defined here (10 is run by hand)"; exit 1 ;;
  esac
done
