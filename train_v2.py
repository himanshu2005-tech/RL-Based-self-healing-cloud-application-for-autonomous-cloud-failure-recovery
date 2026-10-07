"""
Train one agent on one v2 scenario.

    python train_v2.py --algo PPO --env Mixed --seed 1

Every agent keeps the checkpoint with the best mean reward on a fixed set of
validation episodes (VAL_SEEDS); evaluation uses a disjoint set (evaluate_v2.TEST_SEEDS).
Outputs: models/v2/<prefix>.zip|.pkl, results/v2/<prefix>_train.csv (per training
episode) and results/v2/<prefix>_val.csv (validation curve).
"""
import argparse
import dataclasses
import os
import random

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from agents.q_learning_agent import QLearningAgent
from configs import MODEL_DIR, RESULT_DIR, SCENARIOS, VAL_SEEDS
from environment.cloud_env_v2 import (
    CloudSelfHealingEnvV2, OBS_CPU, OBS_MEM, OBS_ERR, OBS_REPLICAS, OBS_PENDING, OBS_SINCE_SCALE,
)



def run_prefix(algo, env_name, seed, cost_aware=True, anti_flapping=True):
    prefix = f"{algo}_{env_name}_seed{seed}"
    if not cost_aware:
        prefix += "_noCost"
    if not anti_flapping:
        prefix += "_noFlap"
    return prefix


def make_q_agent(max_replicas):
    # Edges are in observation units (cpu/2, mem/1.2, replicas/max, ...)
    cpu_edges = np.array([0.4, 0.55, 0.67, 0.8, 1.0, 1.3]) / 2.0
    mem_edges = np.array([0.6, 0.8, 0.9]) / 1.2
    replica_edges = (np.arange(1, max_replicas) + 0.5) / max_replicas
    bins = [cpu_edges, mem_edges, np.array([0.05]), replica_edges,
            np.array([0.5 / max_replicas]), np.array([0.99])]
    feature_idx = [OBS_CPU, OBS_MEM, OBS_ERR, OBS_REPLICAS, OBS_PENDING, OBS_SINCE_SCALE]
    return QLearningAgent(action_space_size=5, learning_rate=0.1, gamma=0.99, epsilon_start=1.0,
                          epsilon_end=0.05, epsilon_decay=0.999, bins=bins, feature_idx=feature_idx)


def validate(policy, config):
    env = CloudSelfHealingEnvV2(config)
    rewards = []
    for s in VAL_SEEDS:
        obs, _ = env.reset(seed=s)
        done, total = False, 0.0
        while not done:
            obs, r, terminated, truncated, _ = env.step(policy(obs))
            total += r
            done = terminated or truncated
        rewards.append(total)
    return float(np.mean(rewards))


class EpisodeStats:
    """Accumulates per-episode metrics from info dicts (one tracker per env index)."""

    def __init__(self):
        self.rows, self.cur = [], {}

    def add(self, i, reward, info, done, timestep):
        c = self.cur.setdefault(i, dict(reward=0.0, steps=0, sla=0, replicas=0.0, crashes=0, flaps=0))
        c["reward"] += float(reward)
        c["steps"] += 1
        c["sla"] += int(info["sla_ok"])
        c["replicas"] += info["replicas"]
        c["crashes"] += int(info["crash"])
        c["flaps"] += int(info["flap"])
        if done:
            self.rows.append({
                "episode": len(self.rows), "timestep": timestep, "reward": c["reward"],
                "sla_violation": 1 - c["sla"] / c["steps"], "mean_replicas": c["replicas"] / c["steps"],
                "crashes": c["crashes"], "flapping_incidents": c["flaps"], "length": c["steps"],
            })
            del self.cur[i]


class TrainCallback(BaseCallback):
    def __init__(self, config, save_path, eval_freq, vecnorm=None):
        super().__init__()
        self.config, self.save_path, self.eval_freq, self.vecnorm = config, save_path, eval_freq, vecnorm
        self.stats, self.val_rows = EpisodeStats(), []
        self.best, self.next_eval = -np.inf, eval_freq

    def _policy(self, obs):
        if self.vecnorm is not None:
            obs = self.vecnorm.normalize_obs(obs)
        return int(self.model.predict(obs, deterministic=True)[0])

    def _on_step(self):
        for i, (r, info, d) in enumerate(zip(self.locals["rewards"], self.locals["infos"], self.locals["dones"])):
            # VecNormalize hands the callback normalized rewards; use the raw ones
            raw = self.vecnorm.get_original_reward()[i] if self.vecnorm is not None else r
            self.stats.add(i, raw, info, d, self.num_timesteps)
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.eval_freq
            score = validate(self._policy, self.config)
            self.val_rows.append({"timestep": self.num_timesteps, "val_reward": score})
            if score > self.best:
                self.best = score
                self.model.save(self.save_path)
                if self.vecnorm is not None:
                    self.vecnorm.save(self.save_path + "_vecnormalize.pkl")
        return True


# Chosen by a validation-only sweep on MemLeak and Mixed (2 seeds each): the stage-B
# settings (lr 5e-4, target update 500, 64x64 net) collapsed on Mixed (best val ~30 vs ~105).
DQN_KWARGS = dict(learning_rate=1e-4, target_update_interval=1000, buffer_size=200_000, batch_size=128,
                  learning_starts=1000, exploration_fraction=0.2, exploration_final_eps=0.02,
                  train_freq=4, gradient_steps=1, policy_kwargs=dict(net_arch=[256, 256]))
DEFAULT_TIMESTEPS = {"PPO": 400_000, "A2C": 400_000, "DQN": 300_000}


def train_sb3(algo, config, seed, timesteps, prefix, algo_kwargs=None):
    save_path = os.path.join(MODEL_DIR, prefix)
    if algo in ("PPO", "A2C"):
        venv = make_vec_env(lambda: CloudSelfHealingEnvV2(config), n_envs=8, seed=seed)
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=0.99)
        if algo == "PPO":
            kwargs = dict(n_steps=512, batch_size=256, ent_coef=0.01, **(algo_kwargs or {}))
            model = PPO("MlpPolicy", venv, verbose=0, seed=seed, **kwargs)
        else:
            kwargs = dict(n_steps=16, ent_coef=0.01, **(algo_kwargs or {}))
            model = A2C("MlpPolicy", venv, verbose=0, seed=seed, **kwargs)
        cb = TrainCallback(config, save_path, eval_freq=20_000, vecnorm=venv)
    else:
        venv = make_vec_env(lambda: CloudSelfHealingEnvV2(config), n_envs=1, seed=seed)
        model = DQN("MlpPolicy", venv, verbose=0, seed=seed, **{**DQN_KWARGS, **(algo_kwargs or {})})
        cb = TrainCallback(config, save_path, eval_freq=10_000)
    model.learn(total_timesteps=timesteps, callback=cb)
    if cb.vecnorm is not None:
        cb.vecnorm.training = False
    return cb.stats.rows, cb.val_rows, cb.best


def train_qlearning(config, seed, episodes, prefix):
    env = CloudSelfHealingEnvV2(config)
    agent = make_q_agent(config.max_replicas)
    stats, val_rows, best = EpisodeStats(), [], -np.inf
    t = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 100_000 + ep)
        done = False
        while not done:
            a = agent.act(obs)
            nxt, r, terminated, truncated, info = env.step(a)
            agent.update(obs, a, r, nxt, terminated)
            done = terminated or truncated
            t += 1
            stats.add(0, r, info, done, t)
            obs = nxt
        agent.decay_epsilon()
        if (ep + 1) % 100 == 0:
            score = validate(lambda o: int(agent.act(o, evaluate=True)), config)
            val_rows.append({"timestep": t, "val_reward": score})
            if score > best:
                best = score
                agent.save(os.path.join(MODEL_DIR, prefix + ".pkl"))
    return stats.rows, val_rows, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["QLearning", "DQN", "PPO", "A2C"], required=True)
    parser.add_argument("--env", choices=list(SCENARIOS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=None, help="defaults: DQN 300k, PPO/A2C 400k")
    parser.add_argument("--episodes", type=int, default=3000, help="Q-learning episodes")
    parser.add_argument("--cost_aware", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--anti_flapping", default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    config = dataclasses.replace(SCENARIOS[args.env], cost_aware=args.cost_aware, anti_flapping=args.anti_flapping)
    prefix = run_prefix(args.algo, args.env, args.seed, args.cost_aware, args.anti_flapping)
    print(f"Training {prefix}", flush=True)

    if args.algo == "QLearning":
        rows, val_rows, best = train_qlearning(config, args.seed, args.episodes, prefix)
    else:
        steps = args.timesteps or DEFAULT_TIMESTEPS[args.algo]
        rows, val_rows, best = train_sb3(args.algo, config, args.seed, steps, prefix)

    pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, f"{prefix}_train.csv"), index=False)
    pd.DataFrame(val_rows).to_csv(os.path.join(RESULT_DIR, f"{prefix}_val.csv"), index=False)
    print(f"Done {prefix}: best validation reward {best:.1f}", flush=True)


if __name__ == "__main__":
    main()
