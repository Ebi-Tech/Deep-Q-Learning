# Gabriel's 10 hyperparameter experiments (PowerShell).
# Usage:  .\run_experiments.ps1                 (all 10, 500000 steps each)
#         .\run_experiments.ps1 -Steps 200000   (shorter runs)
#         .\run_experiments.ps1 -Only 3,4       (just experiments 3 and 4)
# Each run saves models\Gabriel\experiment_<n>.pth and logs\Gabriel\experiment_<n>.csv
param(
    [int]$Steps = 500000,
    [int[]]$Only = @()
)

# n, learning_rate, gamma, batch_size, epsilon_start, epsilon_end, epsilon_decay_steps
$experiments = @(
    @(1,  "1e-4", "0.99",  32, "1.0", "0.01", 100000),  # baseline
    @(2,  "5e-5", "0.99",  32, "1.0", "0.01", 100000),  # lower learning rate
    @(3,  "5e-4", "0.99",  32, "1.0", "0.01", 100000),  # higher learning rate
    @(4,  "1e-4", "0.95",  32, "1.0", "0.01", 100000),  # lower gamma
    @(5,  "1e-4", "0.999", 32, "1.0", "0.01", 100000),  # higher gamma
    @(6,  "1e-4", "0.99",  64, "1.0", "0.01", 100000),  # bigger batch
    @(7,  "1e-4", "0.99",  16, "1.0", "0.01", 100000),  # smaller batch
    @(8,  "1e-4", "0.99",  32, "1.0", "0.10", 100000),  # higher final epsilon
    @(9,  "1e-4", "0.99",  32, "1.0", "0.01", 250000),  # slower epsilon decay
    @(10, "1e-4", "0.99",  32, "0.5", "0.01", 50000)    # less initial exploration, faster decay
)

foreach ($e in $experiments) {
    if ($Only.Count -gt 0 -and $Only -notcontains $e[0]) { continue }
    Write-Host "=== Experiment $($e[0]) ===" -ForegroundColor Cyan
    uv run train.py --member Gabriel --experiment $e[0] --total_steps $Steps `
        --learning_rate $e[1] --gamma $e[2] --batch_size $e[3] `
        --epsilon_start $e[4] --epsilon_end $e[5] --epsilon_decay_steps $e[6]
}
