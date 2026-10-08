"""
Grid-search the rule-based baselines per v2 scenario on the validation episodes
(the same ones the RL agents use for checkpoint selection), so every method gets the
same tuning data. Writes results/v2/tuned_baselines.json.

    python tune_baselines.py [--jobs 6]
"""
import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from agents.baselines import HPAAgent, ThresholdAgent
from configs import RESULT_DIR, SCENARIOS, VAL_SEEDS, for_split
from environment.cloud_env_v2 import CloudSelfHealingEnvV2

TUNED_FILE = os.path.join(RESULT_DIR, "tuned_baselines.json")

GRIDS = {
    "Threshold": dict(up=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75], down=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
                      cooldown=[0, 2, 4, 8, 12, 16], mem=[0.6, 0.65, 0.7, 0.8, 0.9], err=[0.01, 0.02, 0.03, 0.05]),
    "HPA+Heal": dict(target_cpu=[0.45, 0.5, 0.55, 0.6, 0.65, 0.7], stabilization=[1, 3, 5, 8, 12, 16, 24],
                     mem_threshold=[0.6, 0.65, 0.7, 0.8, 0.9], err_threshold=[0.01, 0.02, 0.03, 0.05]),
}


def make_agent(kind, params, max_replicas):
    if kind == "Threshold":
        return ThresholdAgent(max_replicas=max_replicas, **params)
    return HPAAgent(max_replicas=max_replicas, heal=True, **params)


def score(kind, params, config, seeds=VAL_SEEDS):
    env = CloudSelfHealingEnvV2(for_split(config, "val"))
    totals = []
    for s in seeds:
        agent = make_agent(kind, params, config.max_replicas)
        obs, _ = env.reset(seed=s)
        done, total = False, 0.0
        while not done:
            obs, r, terminated, truncated, _ = env.step(agent.act(obs))
            total += r
            done = terminated or truncated
        totals.append(total)
    return float(np.mean(totals))


def tune_one(job):
    env_name, kind, values = job
    params = dict(zip(GRIDS[kind].keys(), values))
    return env_name, kind, params, score(kind, params, SCENARIOS[env_name])


def edge_params(kind, params):
    """Parameters that landed on the edge of the grid (a sign the grid may be too narrow)."""
    return [k for k, v in params.items() if len(GRIDS[kind][k]) > 1 and v in (GRIDS[kind][k][0], GRIDS[kind][k][-1])]


def load_tuned():
    with open(TUNED_FILE) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--envs", nargs="+", default=list(SCENARIOS))
    args = parser.parse_args()
    jobs = [(e, kind, v) for e in args.envs for kind, grid in GRIDS.items() for v in itertools.product(*grid.values())]
    results = {e: {} for e in args.envs}
    with ProcessPoolExecutor(args.jobs) as pool:
        for env_name, kind, params, s in pool.map(tune_one, jobs, chunksize=50):
            best = results[env_name].get(kind)
            if best is None or s > best["val_reward"]:
                results[env_name][kind] = {"params": params, "val_reward": s}
    os.makedirs(RESULT_DIR, exist_ok=True)
    # Keep previously tuned scenarios that were not re-tuned this time
    merged = {**(load_tuned() if os.path.exists(TUNED_FILE) else {}), **results}
    with open(TUNED_FILE, "w") as f:
        json.dump(merged, f, indent=2)
    for env_name, r in results.items():
        for kind, v in r.items():
            edges = edge_params(kind, v["params"])
            print(f"{env_name:11s} {kind:10s} val {v['val_reward']:7.1f}  {v['params']}"
                  + (f"  [at grid edge: {', '.join(edges)}]" if edges else ""))


if __name__ == "__main__":
    main()
