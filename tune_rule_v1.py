"""
Grid-search the v1 RuleBasedAgent thresholds per traffic profile on validation episodes
(seeds 500-519, disjoint from evaluate.py's test episodes). Writes results/v1_tuned_rule.json.

    python tune_rule_v1.py
"""
import itertools
import json

import numpy as np

from agents.rule_based_agent import RuleBasedAgent
from configs import CONFIGS, make_config
from environment.cloud_env import CloudSelfHealingEnv

VAL_SEEDS = range(500, 520)
TUNED_FILE = "results/v1_tuned_rule.json"
GRID = dict(restart_cpu=[0.5, 0.6, 0.7, 0.8, 0.9, 1.01], restart_err=[0.1, 0.3, 1.01],
            scale_up_cpu=[0.6, 0.75, 0.9, 1.01], scale_down=[0.0, 0.1, 0.3], cache_rt=[3.0, 6.0, 10.1])


def score(thresholds, env):
    agent = RuleBasedAgent(**thresholds)
    totals = []
    for s in VAL_SEEDS:
        state, _ = env.reset(seed=s)
        done, total = False, 0.0
        while not done:
            state, r, terminated, truncated, _ = env.step(agent.act(state))
            total += r
            done = terminated or truncated
        totals.append(total)
    return float(np.mean(totals))


def main():
    out = {}
    for name in CONFIGS:
        env = CloudSelfHealingEnv(config=make_config(name))
        best = max(((score(dict(zip(GRID, v)), env), dict(zip(GRID, v))) for v in itertools.product(*GRID.values())),
                   key=lambda x: x[0])
        out[name] = {"thresholds": best[1], "val_reward": best[0]}
        print(f"{name:14s} val {best[0]:7.2f}  {best[1]}", flush=True)
    with open(TUNED_FILE, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
