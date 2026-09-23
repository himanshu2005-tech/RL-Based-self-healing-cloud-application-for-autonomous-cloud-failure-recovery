param (
    [int[]]$Seeds = @(1, 2, 3),
    [string[]]$Configs = @("LowTraffic", "HighTraffic", "BurstyTraffic")
)

Write-Host "========================================"
Write-Host "Running RL Self-Healing Cloud Experiment"
Write-Host "========================================"

# Run Calibration if missing
if (-Not (Test-Path "data_analysis\calibration_summary.json")) {
    Write-Host "Running data calibration..."
    python data_analysis\calibrate_from_gcd.py
}

# Main Training Loop
foreach ($seed in $Seeds) {
    foreach ($config in $Configs) {
        Write-Host "`n--- Training Seed: $seed | Config: $config ---"
        
        Write-Host "-> Q-Learning"
        python train_qlearning.py --seed $seed --config $config --episodes 500
        
        Write-Host "-> DQN"
        python train_dqn.py --seed $seed --config $config --timesteps 20000
        
        Write-Host "-> PPO"
        python train_ppo.py --seed $seed --config $config --timesteps 20000
    }
}

# Ablation Study Loop (DQN on LowTraffic only)
Write-Host "`n========================================"
Write-Host "Running Ablation Study for DQN"
Write-Host "========================================"
foreach ($seed in $Seeds) {
    Write-Host "-> No Cost Aware"
    python train_dqn.py --seed $seed --config LowTraffic --no-cost_aware --timesteps 20000
    
    Write-Host "-> No Anti Flapping"
    python train_dqn.py --seed $seed --config LowTraffic --no-anti_flapping --timesteps 20000
    
    Write-Host "-> Plain Baseline (No Cost, No Flapping)"
    python train_dqn.py --seed $seed --config LowTraffic --no-cost_aware --no-anti_flapping --timesteps 20000
}

Write-Host "`n========================================"
Write-Host "Evaluating Models and Generating Plots"
Write-Host "========================================"
python evaluate.py

Write-Host "`nExperiments completed! Results are in 'results/' and plots in 'plots/'"
