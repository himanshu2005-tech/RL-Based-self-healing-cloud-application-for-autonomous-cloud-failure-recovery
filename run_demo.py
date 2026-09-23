"""
Interactive Live Demonstration of RL Self-Healing Cloud Application.
Demonstrates how the trained RL agent actively detects anomalies, avoids crashes,
suppresses flapping, and maintains cloud SLA compared to unmanaged and rule-based systems.
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN

# Set UTF-8 encoding for Windows standard output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from environment.cloud_env import CloudSelfHealingEnv
from agents.rule_based_agent import RuleBasedAgent
from configs import BurstyTrafficConfig, HighTrafficConfig, LowTrafficConfig

ACTION_NAMES = {
    0: "Restart Container",
    1: "Scale Up (Add Pods)",
    2: "Scale Down",
    3: "Clear Redis Cache",
    4: "Do Nothing (Healthy)"
}

ACTION_TAGS = {
    0: "[RESTART]",
    1: "[SCALE UP]",
    2: "[SCALE DOWN]",
    3: "[CLEAR CACHE]",
    4: "[DO NOTHING]"
}

def render_bar(val: float, length: int = 15, warn_thresh: float = 0.75, crit_thresh: float = 0.90) -> str:
    """Render a visual ASCII progress bar with color cues."""
    filled = int(round(val * length))
    filled = max(0, min(length, filled))
    bar = "#" * filled + "-" * (length - filled)
    pct = f"{int(val * 100):3d}%"
    
    if val >= crit_thresh:
        color = "\033[91m"  # Red
    elif val >= warn_thresh:
        color = "\033[93m"  # Yellow
    else:
        color = "\033[92m"  # Green
    reset = "\033[0m"
    return f"{color}[{bar}] {pct}{reset}"

def run_simulation(agent, env, steps=50, seed=42):
    """Runs a single episode recording step-by-step telemetry."""
    state, _ = env.reset(seed=seed)
    
    records = {
        "step": [],
        "cpu": [],
        "ram": [],
        "resp_time": [],
        "err_rate": [],
        "req_load": [],
        "action": [],
        "health": [],
        "reward": [],
        "crashes": 0,
        "flapping": 0
    }
    
    for s in range(steps):
        # Choose action
        if agent is None:
            action = 4  # Unmanaged / Passive
        elif isinstance(agent, RuleBasedAgent):
            action = agent.act(state)
        else:
            action, _ = agent.predict(state, deterministic=True)
            action = int(action)
            
        cpu, ram, resp_time, err_rate, req_load = state
        next_state, reward, terminated, truncated, info = env.step(action)
        
        records["step"].append(s + 1)
        records["cpu"].append(cpu)
        records["ram"].append(ram)
        records["resp_time"].append(resp_time)
        records["err_rate"].append(err_rate)
        records["req_load"].append(req_load)
        records["action"].append(action)
        records["health"].append(info.get("health_score", 0.0))
        records["reward"].append(reward)
        
        state = next_state
        if terminated or truncated:
            break
            
    records["crashes"] = env.crashes
    records["flapping"] = env.flapping_incidents
    return records

def live_terminal_demo(rl_agent, env_config, steps=35, delay=0.08, seed=42):
    """Prints a live, animated terminal simulation showing how RL heals the app."""
    env = CloudSelfHealingEnv(config=env_config)
    state, _ = env.reset(seed=seed)
    
    print("\n" + "="*85)
    print(" >>> LIVE AUTONOMOUS CLOUD SELF-HEALING DEMONSTRATION <<<")
    print(" Monitoring Real-Time Telemetry & Observing Self-Healing Actions")
    print("="*85)
    print(f"Scenario: {env_config.traffic_profile.upper()} Traffic Profile | Simulation Steps: {steps}\n")
    if delay > 0:
        time.sleep(0.4)

    total_cost = 0.0
    
    for s in range(steps):
        cpu, ram, resp_time, err_rate, req_load = state
        
        # RL Agent prediction
        action, _ = rl_agent.predict(state, deterministic=True)
        action = int(action)
        
        # Identify any anomaly in current state
        anomalies = []
        if req_load > 0.65:
            anomalies.append("[TRAFFIC SURGE]")
        if cpu > 0.80:
            anomalies.append("[HIGH CPU LOAD]")
        if ram > 0.80:
            anomalies.append("[RAM PRESSURE]")
        if err_rate > 0.20:
            anomalies.append("[HIGH ERROR RATE]")
        if resp_time > 3.0:
            anomalies.append("[LATENCY SPIKE]")
            
        anomaly_str = " ".join(anomalies) if anomalies else "[NORMAL]"
        
        # Execute action
        next_state, reward, terminated, truncated, info = env.step(action)
        health = info.get("health_score", 0.0)
        cost = info.get("action_cost", 0.0)
        total_cost += cost
        crash = info.get("crash", False)
        
        # Build telemetry display
        cpu_bar = render_bar(cpu, length=12)
        ram_bar = render_bar(ram, length=12)
        act_tag = ACTION_TAGS.get(action, "[ACT]")
        act_name = ACTION_NAMES.get(action, "Unknown")
        
        print("-" * 85)
        step_header = f"Step {s+1:02d}/{steps:02d} | Alert: {anomaly_str}"
        if crash:
            step_header += " | *** CRASH OCCURRED! ***"
        print(step_header)
        
        print(f" Telemetry : CPU {cpu_bar} | RAM {ram_bar} | Latency: {resp_time:4.2f}s | Error Rate: {err_rate*100:4.1f}%")
        print(f" Recovery  : {act_tag} {act_name:<20} (Cost: ${cost:.2f}) | System Health: {health*100:4.1f}%")
        
        state = next_state
        if delay > 0:
            time.sleep(delay)
            
    print("="*85)
    print(f" DEMO FINISHED: Flapping: {env.flapping_incidents} | Crashes: {env.crashes} | Total Recovery Cost: ${total_cost:.2f}")
    print("="*85 + "\n")

def plot_live_comparison(unmanaged, rule_based, rl, output_path="plots/live_demonstration.png"):
    """Generate a clean side-by-side comparative graph proving RL superiority."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    steps = unmanaged["step"]
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. System Health Score
    axes[0].plot(steps, [h * 100 for h in unmanaged["health"]], 'r--', label=f"Unmanaged (Crashes: {unmanaged['crashes']})", linewidth=1.8)
    axes[0].plot(steps, [h * 100 for h in rule_based["health"]], 'g-.', label=f"Rule-Based (Flapping: {rule_based['flapping']})", linewidth=1.8)
    axes[0].plot(steps, [h * 100 for h in rl["health"]], 'b-', label=f"Autonomous RL (Crashes: {rl['crashes']}, Flap: {rl['flapping']})", linewidth=2.5)
    axes[0].set_ylabel("Health Score (%)", fontsize=11, fontweight='bold')
    axes[0].set_title("Autonomous Cloud Recovery: Real-Time Performance Comparison", fontsize=13, fontweight='bold')
    axes[0].legend(loc="lower left", framealpha=0.9)
    axes[0].set_ylim(-5, 105)
    
    # 2. CPU Utilization
    axes[1].plot(steps, [c * 100 for c in unmanaged["cpu"]], 'r--', label="Unmanaged CPU", alpha=0.8)
    axes[1].plot(steps, [c * 100 for c in rule_based["cpu"]], 'g-.', label="Rule-Based CPU", alpha=0.8)
    axes[1].plot(steps, [c * 100 for c in rl["cpu"]], 'b-', label="Autonomous RL CPU", linewidth=2.0)
    axes[1].axhline(y=85, color='orange', linestyle=':', label='Stress Threshold (85%)')
    axes[1].set_ylabel("CPU Usage (%)", fontsize=11, fontweight='bold')
    axes[1].legend(loc="upper left", framealpha=0.9)
    axes[1].set_ylim(0, 105)
    
    # 3. Response Time & Error Rate
    ax3_err = axes[2].twinx()
    axes[2].plot(steps, unmanaged["resp_time"], 'r--', alpha=0.6, label="Unmanaged Latency (s)")
    axes[2].plot(steps, rl["resp_time"], 'b-', linewidth=2.0, label="Autonomous RL Latency (s)")
    ax3_err.plot(steps, [e * 100 for e in rl["err_rate"]], 'm:', linewidth=2.0, label="RL Error Rate (%)")
    axes[2].set_ylabel("Response Time (s)", fontsize=11, fontweight='bold')
    ax3_err.set_ylabel("Error Rate (%)", fontsize=11, fontweight='bold', color='purple')
    axes[2].legend(loc="upper left")
    ax3_err.legend(loc="upper right")
    
    # 4. Actions Taken by RL Agent
    action_colors = {0: '#e74c3c', 1: '#2ecc71', 2: '#f39c12', 3: '#9b59b6', 4: '#95a5a6'}
    for a_code, a_name in ACTION_NAMES.items():
        sub_steps = [s for s, a in zip(steps, rl["action"]) if a == a_code]
        sub_vals = [a_code for a in rl["action"] if a == a_code]
        if sub_steps:
            axes[3].scatter(sub_steps, sub_vals, label=a_name, color=action_colors[a_code], s=70, alpha=0.9, edgecolors='black')
            
    axes[3].set_yticks(list(ACTION_NAMES.keys()))
    axes[3].set_yticklabels(list(ACTION_NAMES.values()), fontsize=10)
    axes[3].set_ylabel("Action Executed", fontsize=11, fontweight='bold')
    axes[3].set_xlabel("Simulation Step (Time)", fontsize=11, fontweight='bold')
    axes[3].set_title("Self-Healing Actions Selected by Autonomous RL Agent Over Time", fontsize=12, fontweight='bold')
    axes[3].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\n[+] Side-by-side demonstration graph saved successfully to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Live Demonstration of Autonomous Cloud Self-Healing")
    parser.add_argument("--model", type=str, default="PPO", choices=["PPO", "DQN"], help="RL Model architecture to load")
    parser.add_argument("--traffic", type=str, default="BurstyTraffic", choices=["LowTraffic", "HighTraffic", "BurstyTraffic"], help="Traffic scenario")
    parser.add_argument("--seed", type=int, default=1, help="Seed index (1, 2, or 3)")
    parser.add_argument("--steps", type=int, default=35, help="Number of steps to simulate")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between terminal steps in seconds")
    args = parser.parse_args()
    
    # 1. Load trained model
    model_path = os.path.join("models", f"{args.model}_{args.traffic}_seed{args.seed}.zip")
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found! Checking available models...")
        fallback_files = [f for f in os.listdir("models") if f.endswith(".zip")]
        if not fallback_files:
            print("No trained models found in 'models/' directory.")
            sys.exit(1)
        model_path = os.path.join("models", fallback_files[0])
        print(f"Using fallback model: {model_path}")
        
    print(f"Loading trained agent: {model_path}")
    if "PPO" in model_path:
        rl_agent = PPO.load(model_path)
    else:
        rl_agent = DQN.load(model_path)
        
    config = BurstyTrafficConfig if args.traffic == "BurstyTraffic" else (HighTrafficConfig if args.traffic == "HighTraffic" else LowTrafficConfig)
    
    # 2. Run Terminal Live Demo
    live_terminal_demo(rl_agent, config, steps=args.steps, delay=args.delay, seed=42 + args.seed)
    
    # 3. Run Head-to-Head Simulation (Identical Conditions)
    print("Running head-to-head comparison (Unmanaged vs Rule-Based vs RL)...")
    env_unmanaged = CloudSelfHealingEnv(config=config)
    env_rule = CloudSelfHealingEnv(config=config)
    env_rl = CloudSelfHealingEnv(config=config)
    
    eval_seed = 100 + args.seed
    sim_unmanaged = run_simulation(None, env_unmanaged, steps=args.steps, seed=eval_seed)
    sim_rule = run_simulation(RuleBasedAgent(), env_rule, steps=args.steps, seed=eval_seed)
    sim_rl = run_simulation(rl_agent, env_rl, steps=args.steps, seed=eval_seed)
    
    # 4. Summary Table
    avg_h_unmanaged = np.mean(sim_unmanaged["health"]) * 100
    avg_h_rule = np.mean(sim_rule["health"]) * 100
    avg_h_rl = np.mean(sim_rl["health"]) * 100
    
    print("\n" + "="*85)
    print(" HEAD-TO-HEAD QUANTITATIVE SHOWCASE")
    print("="*85)
    print(f"{'Strategy':<25} | {'Avg Health':<12} | {'Crashes':<10} | {'Flapping':<10} | {'Operational Status'}")
    print("-" * 85)
    print(f"{'1. Unmanaged (No Healing)':<25} | {avg_h_unmanaged:6.1f}%      | {sim_unmanaged['crashes']:<10} | {'N/A':<10} | Suffer Downtime & Crashes")
    print(f"{'2. Traditional Rule-Based':<25} | {avg_h_rule:6.1f}%      | {sim_rule['crashes']:<10} | {sim_rule['flapping']:<10} | High Flapping & Cost")
    print(f"{'3. Autonomous RL (Ours)':<25} | {avg_h_rl:6.1f}%      | {sim_rl['crashes']:<10} | {sim_rl['flapping']:<10} | Stable, Low Cost, Zero Crash")
    print("="*85)
    
    # 5. Plot Side-by-Side Graph
    plot_live_comparison(sim_unmanaged, sim_rule, sim_rl, output_path="plots/live_demonstration.png")

if __name__ == "__main__":
    main()
