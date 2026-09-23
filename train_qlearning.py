import argparse
import os
import random
import numpy as np
import pandas as pd
from environment.cloud_env import CloudSelfHealingEnv
from agents.q_learning_agent import QLearningAgent
from configs import CONFIGS, EnvConfig

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=str, choices=list(CONFIGS.keys()), default="LowTraffic")
    parser.add_argument("--cost_aware", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--anti_flapping", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    set_seed(args.seed)

    base_config = CONFIGS[args.config]
    # Override ablation flags
    base_config.cost_aware = args.cost_aware
    base_config.anti_flapping = args.anti_flapping

    env = CloudSelfHealingEnv(config=base_config)
    # We don't seed the env here directly with a global seed if we want it to be stochastic but reproducible per step,
    # but Gymnasium requires seed passed to reset.

    agent = QLearningAgent(
        action_space_size=env.action_space.n,
        num_bins=5,
        learning_rate=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995
    )

    results = []
    
    print(f"Training Q-Learning on {args.config} | Seed {args.seed} | Cost-Aware {args.cost_aware} | Anti-Flapping {args.anti_flapping}")

    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed + episode)
        terminated = False
        truncated = False
        
        episode_reward = 0
        episode_health = 0
        steps = 0
        
        while not (terminated or truncated):
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            agent.update(state, action, reward, next_state)
            
            state = next_state
            episode_reward += reward
            episode_health += info["health_score"]
            steps += 1
            
        agent.decay_epsilon()
        
        results.append({
            "episode": episode,
            "reward": episode_reward,
            "avg_health": episode_health / steps,
            "crashes": env.crashes,
            "flapping_incidents": env.flapping_incidents
        })
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}/{args.episodes} | Reward: {episode_reward:.2f} | EPS: {agent.epsilon:.2f} | Crashes: {env.crashes} | Flapping: {env.flapping_incidents}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    prefix = f"QLearning_{args.config}_seed{args.seed}"
    if not args.cost_aware:
        prefix += "_noCost"
    if not args.anti_flapping:
        prefix += "_noFlap"
        
    model_path = os.path.join("models", f"{prefix}.pkl")
    agent.save(model_path)
    
    csv_path = os.path.join("results", f"{prefix}.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Saved model to {model_path} and results to {csv_path}")

if __name__ == "__main__":
    main()
