import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from stable_baselines3 import DQN, PPO

from environment.cloud_env import CloudSelfHealingEnv
from agents.q_learning_agent import QLearningAgent
from agents.rule_based_agent import RuleBasedAgent
from configs import CONFIGS, make_config

# Configuration
EVAL_EPISODES = 10
SEEDS = [1, 2, 3]
ALGORITHMS = ["RuleBased", "QLearning", "DQN", "PPO"]
ENV_VARIANTS = ["LowTraffic", "HighTraffic", "BurstyTraffic"]
ACTIONS = ["Restart", "Scale Up", "Scale Down", "Clear Cache", "Do Nothing"]

def evaluate_agent(agent, env, num_episodes=10, seed_offset=1000):
    rewards = []
    healths = []
    crashes = []
    flapping = []
    action_counts = {0:0, 1:0, 2:0, 3:0, 4:0}
    
    for i in range(num_episodes):
        state, _ = env.reset(seed=seed_offset + i)
        done = False
        
        ep_reward = 0
        ep_health = 0
        steps = 0
        
        while not done:
            if isinstance(agent, RuleBasedAgent) or isinstance(agent, QLearningAgent):
                action = agent.act(state, evaluate=True) if isinstance(agent, QLearningAgent) else agent.act(state)
            else:
                action, _ = agent.predict(state, deterministic=True)
                
            action_counts[int(action)] += 1
            
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ep_reward += reward
            ep_health += info.get("health_score", 0)
            steps += 1
            
        rewards.append(ep_reward)
        healths.append(ep_health / max(1, steps))
        crashes.append(env.crashes)
        flapping.append(env.flapping_incidents)
        
    # Normalize action distribution
    total_actions = sum(action_counts.values())
    if total_actions > 0:
        action_dist = {k: v/total_actions for k, v in action_counts.items()}
    else:
        action_dist = action_counts
        
    return {
        "reward_mean": np.mean(rewards),
        "reward_std": np.std(rewards),
        "health_mean": np.mean(healths),
        "health_std": np.std(healths),
        "crashes_mean": np.mean(crashes),
        "crashes_std": np.std(crashes),
        "flapping_mean": np.mean(flapping),
        "flapping_std": np.std(flapping),
        "action_dist": action_dist
    }

def main():
    os.makedirs("plots", exist_ok=True)
    results_summary = []
    
    print("Evaluating models...")
    for env_name in ENV_VARIANTS:
        config = CONFIGS[env_name]
        env = CloudSelfHealingEnv(config=config)
        
        for algo in ALGORITHMS:
            # We aggregate across seeds
            algo_rewards = []
            algo_healths = []
            algo_crashes = []
            algo_flapping = []
            algo_action_dist = {0:0, 1:0, 2:0, 3:0, 4:0}
            
            seeds_to_eval = SEEDS if algo != "RuleBased" else [1]
            valid_seeds = 0
            
            for seed in seeds_to_eval:
                agent = None
                if algo == "RuleBased":
                    agent = RuleBasedAgent()
                elif algo == "QLearning":
                    model_path = f"models/QLearning_{env_name}_seed{seed}.pkl"
                    if os.path.exists(model_path):
                        agent = QLearningAgent(action_space_size=5)
                        agent.load(model_path)
                elif algo == "DQN":
                    model_path = f"models/DQN_{env_name}_seed{seed}.zip"
                    if os.path.exists(model_path):
                        agent = DQN.load(model_path)
                elif algo == "PPO":
                    model_path = f"models/PPO_{env_name}_seed{seed}.zip"
                    if os.path.exists(model_path):
                        agent = PPO.load(model_path)
                        
                if agent is not None:
                    res = evaluate_agent(agent, env, num_episodes=EVAL_EPISODES, seed_offset=1000 + seed*100)
                    algo_rewards.append(res["reward_mean"])
                    algo_healths.append(res["health_mean"])
                    algo_crashes.append(res["crashes_mean"])
                    algo_flapping.append(res["flapping_mean"])
                    for k, v in res["action_dist"].items():
                        algo_action_dist[k] += v
                    valid_seeds += 1
            
            if valid_seeds > 0:
                summary = {
                    "Environment": env_name,
                    "Algorithm": algo,
                    "Reward (Mean \u00B1 Std)": f"{np.mean(algo_rewards):.2f} \u00B1 {np.std(algo_rewards):.2f}",
                    "Health (Mean \u00B1 Std)": f"{np.mean(algo_healths):.3f} \u00B1 {np.std(algo_healths):.3f}",
                    "Crashes (Mean)": f"{np.mean(algo_crashes):.2f}",
                    "Flapping (Mean)": f"{np.mean(algo_flapping):.2f}"
                }
                results_summary.append(summary)
                
                # Plot action distribution
                avg_action_dist = [algo_action_dist[i] / valid_seeds for i in range(5)]
                plt.figure(figsize=(8, 5))
                sns.barplot(x=ACTIONS, y=avg_action_dist)
                plt.title(f"Action Distribution - {algo} on {env_name}")
                plt.ylabel("Frequency")
                plt.ylim(0, 1)
                plt.savefig(f"plots/ActionDist_{algo}_{env_name}.png")
                plt.close()

    df_summary = pd.DataFrame(results_summary)
    print("\n--- Evaluation Results ---")
    print(df_summary.to_string(index=False))
    df_summary.to_csv("results/evaluation_summary.csv", index=False)
    
    # --- Ablation Study Evaluation ---
    print("\n--- Ablation Study (DQN on LowTraffic) ---")
    ablation_results = []
    base_env_name = "LowTraffic"
    ablation_variants = [
        ("", True, True),          # Full
        ("_noCost", False, True),  # No Cost
        ("_noFlap", True, False),  # No Flapping
        ("_noCost_noFlap", False, False) # Plain
    ]
    
    for suffix, cost_aware, anti_flapping in ablation_variants:
        variant_rewards = []
        variant_flapping = []
        valid_seeds = 0
        
        # We need an env with the right config for evaluation logic
        ablation_config = make_config(base_env_name, cost_aware=cost_aware, anti_flapping=anti_flapping)
        ablation_env = CloudSelfHealingEnv(config=ablation_config)
        
        for seed in SEEDS:
            model_path = f"models/DQN_{base_env_name}_seed{seed}{suffix}.zip"
            if os.path.exists(model_path):
                agent = DQN.load(model_path)
                res = evaluate_agent(agent, ablation_env, num_episodes=EVAL_EPISODES, seed_offset=2000 + seed*100)
                variant_rewards.append(res["reward_mean"])
                variant_flapping.append(res["flapping_mean"])
                valid_seeds += 1
                
        if valid_seeds > 0:
            ablation_results.append({
                "Variant": f"Cost={cost_aware}, Flapping={anti_flapping}",
                "Reward": f"{np.mean(variant_rewards):.2f} \u00B1 {np.std(variant_rewards):.2f}",
                "Flapping Incidents": f"{np.mean(variant_flapping):.2f} \u00B1 {np.std(variant_flapping):.2f}"
            })
            
    if ablation_results:
        df_ablation = pd.DataFrame(ablation_results)
        print(df_ablation.to_string(index=False))
        df_ablation.to_csv("results/ablation_summary.csv", index=False)
    else:
        print("Ablation models not found. Ensure DQN was trained with --no-cost_aware and --no-anti_flapping flags.")

if __name__ == "__main__":
    main()
