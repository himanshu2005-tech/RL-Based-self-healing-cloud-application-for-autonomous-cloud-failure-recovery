"""
Evaluate every agent on the v2 scenarios using the same test episodes.

    python evaluate_v2.py [--seeds 1 2 3] [--envs Mixed ...]

Writes results/v2/evaluation_episodes.csv (one row per agent/seed/episode) and
results/v2/evaluation_summary.csv. Std is over all test episodes pooled across
training seeds; "Seed Std" is the std of the per-seed means.
"""
import argparse
import os

import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.baselines import HoldAgent, HPAAgent
from configs import SCENARIOS
from environment.cloud_env_v2 import CloudSelfHealingEnvV2, ACTION_NAMES
from train_v2 import MODEL_DIR, RESULT_DIR, make_q_agent, run_prefix

TEST_SEEDS = range(10_000, 10_050)
BASELINES = ["Hold", "HPA", "HPA+Heal"]
LEARNED = ["QLearning", "DQN", "PPO"]


def load_policy(algo, env_name, seed, config):
    """Returns obs -> action, or None if the model file is missing."""
    if algo == "Hold":
        return HoldAgent().act
    if algo in ("HPA", "HPA+Heal"):
        return HPAAgent(max_replicas=config.max_replicas, heal=(algo == "HPA+Heal")).act
    path = os.path.join(MODEL_DIR, run_prefix(algo, env_name, seed, config.cost_aware, config.anti_flapping))
    if algo == "QLearning":
        if not os.path.exists(path + ".pkl"):
            return None
        agent = make_q_agent(config.max_replicas)
        agent.load(path + ".pkl")
        return lambda obs: int(agent.act(obs, evaluate=True))
    if not os.path.exists(path + ".zip"):
        return None
    model = (PPO if algo == "PPO" else DQN).load(path + ".zip", device="cpu")
    if os.path.exists(path + "_vecnormalize.pkl"):
        vecnorm = VecNormalize.load(path + "_vecnormalize.pkl", DummyVecEnv([lambda: CloudSelfHealingEnvV2(config)]))
        vecnorm.training = False
        return lambda obs: int(model.predict(vecnorm.normalize_obs(obs), deterministic=True)[0])
    return lambda obs: int(model.predict(obs, deterministic=True)[0])


def run_episodes(make_policy, config):
    env = CloudSelfHealingEnvV2(config)
    rows = []
    for s in TEST_SEEDS:
        policy = make_policy()  # fresh agent state (HPA history) per episode
        obs, _ = env.reset(seed=s)
        done = False
        reward, sla, replicas, steps = 0.0, 0, 0.0, 0
        actions = np.zeros(5)
        while not done:
            a = policy(obs)
            actions[a] += 1
            obs, r, terminated, truncated, info = env.step(a)
            reward += r
            sla += int(info["sla_ok"])
            replicas += info["replicas"]
            steps += 1
            done = terminated or truncated
        rows.append({
            "episode_seed": s, "reward": reward, "sla_violation": 1 - sla / steps,
            "mean_replicas": replicas / steps, "crashes": env.crashes,
            "flapping_incidents": env.flapping_incidents, "length": steps,
            **{f"act_{n}": actions[i] / steps for i, n in enumerate(ACTION_NAMES)},
        })
    return rows


def summarize(df):
    out = []
    for (env_name, algo), g in df.groupby(["Environment", "Algorithm"], sort=False):
        seed_means = g.groupby("train_seed")["reward"].mean()
        out.append({
            "Environment": env_name,
            "Algorithm": algo,
            "Seeds": g["train_seed"].nunique(),
            "Reward (Mean ± Std)": f"{g.reward.mean():.1f} ± {g.reward.std(ddof=0):.1f}",
            "Seed Std": f"{seed_means.std(ddof=0):.1f}",
            "SLA Violation %": f"{100 * g.sla_violation.mean():.1f}",
            "Replicas": f"{g.mean_replicas.mean():.2f}",
            "Crash Rate": f"{g.crashes.mean():.2f}",
            "Flaps / Ep": f"{g.flapping_incidents.mean():.2f}",
            "Actions R/U/D/C/H %": "/".join(f"{100 * g[f'act_{n}'].mean():.0f}" for n in ACTION_NAMES),
        })
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--envs", nargs="+", default=list(SCENARIOS))
    args = parser.parse_args()

    rows = []
    for env_name in args.envs:
        config = SCENARIOS[env_name]
        for algo in BASELINES + LEARNED:
            # Baselines are deterministic, so one "seed" is enough
            for seed in ([0] if algo in BASELINES else args.seeds):
                policy = load_policy(algo, env_name, seed, config)
                if policy is None:
                    print(f"  missing model: {algo} {env_name} seed {seed}")
                    continue
                if algo in BASELINES:
                    make_policy = lambda: load_policy(algo, env_name, seed, config)
                else:
                    make_policy = lambda: policy
                for r in run_episodes(make_policy, config):
                    rows.append({"Environment": env_name, "Algorithm": algo, "train_seed": seed, **r})
        print(f"evaluated {env_name}", flush=True)

    os.makedirs(RESULT_DIR, exist_ok=True)
    episodes = pd.DataFrame(rows)
    episodes.to_csv(os.path.join(RESULT_DIR, "evaluation_episodes.csv"), index=False)
    summary = summarize(episodes)
    summary.to_csv(os.path.join(RESULT_DIR, "evaluation_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
