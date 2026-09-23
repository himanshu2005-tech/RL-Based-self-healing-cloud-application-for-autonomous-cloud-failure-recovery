import argparse
import os
import random
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

from environment.cloud_env import CloudSelfHealingEnv
from configs import CONFIGS, EnvConfig

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

class MetricsLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MetricsLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_healths = []
        self.episode_crashes = []
        self.episode_flapping = []
        
        self.current_reward = 0
        self.current_health = 0
        self.current_steps = 0
        self.current_crashes = 0
        self.current_flapping = 0

    def _on_step(self) -> bool:
        # Assuming single environment vectorization
        info = self.locals["infos"][0]
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]
        
        self.current_reward += reward
        self.current_health += info.get("health_score", 0)
        self.current_steps += 1
        
        if done:
            # We access the underlying env's accumulated variables
            # In DummyVecEnv, we can access the env directly
            env = self.training_env.envs[0].unwrapped
            self.episode_rewards.append(self.current_reward)
            self.episode_healths.append(self.current_health / max(1, self.current_steps))
            self.episode_crashes.append(env.crashes)
            self.episode_flapping.append(env.flapping_incidents)
            
            self.current_reward = 0
            self.current_health = 0
            self.current_steps = 0
            
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=str, choices=list(CONFIGS.keys()), default="LowTraffic")
    parser.add_argument("--cost_aware", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--anti_flapping", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--timesteps", type=int, default=50000)
    args = parser.parse_args()

    set_seed(args.seed)

    base_config = CONFIGS[args.config]
    base_config.cost_aware = args.cost_aware
    base_config.anti_flapping = args.anti_flapping

    def make_env():
        return CloudSelfHealingEnv(config=base_config)
        
    env = make_vec_env(make_env, n_envs=1, seed=args.seed)

    print(f"Training DQN on {args.config} | Seed {args.seed} | Cost-Aware {args.cost_aware} | Anti-Flapping {args.anti_flapping}")

    model = DQN("MlpPolicy", env, verbose=0, seed=args.seed)
    
    callback = MetricsLoggingCallback()
    model.learn(total_timesteps=args.timesteps, callback=callback)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    prefix = f"DQN_{args.config}_seed{args.seed}"
    if not args.cost_aware:
        prefix += "_noCost"
    if not args.anti_flapping:
        prefix += "_noFlap"
        
    model_path = os.path.join("models", f"{prefix}")
    model.save(model_path)
    
    # Save results
    results = []
    for i in range(len(callback.episode_rewards)):
        results.append({
            "episode": i,
            "reward": callback.episode_rewards[i],
            "avg_health": callback.episode_healths[i],
            "crashes": callback.episode_crashes[i],
            "flapping_incidents": callback.episode_flapping[i]
        })
        
    csv_path = os.path.join("results", f"{prefix}.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Saved model to {model_path}.zip and results to {csv_path}")

if __name__ == "__main__":
    main()
