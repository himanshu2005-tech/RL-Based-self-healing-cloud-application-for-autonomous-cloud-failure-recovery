"""
Train every algorithm on every v2 scenario and seed in parallel, tune the rule
baselines on the validation episodes, then evaluate.

    python run_v2_experiments.py --seeds 1 2 3 4 5 --jobs 6 --ablation-env Mixed

--ablation-env adds PPO runs without the action-cost and/or flapping terms.
--eval-seeds lets training cover only new seeds while evaluating all of them.
"""
import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from configs import SCENARIOS, SYNTHETIC_SCENARIOS

LOG_DIR = os.path.join("results", "v2", "logs")


def run(cmd, log_name):
    with open(os.path.join(LOG_DIR, log_name), "w") as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f"{'ok' if code == 0 else 'FAILED'}  {log_name}", flush=True)
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--envs", nargs="+", default=SYNTHETIC_SCENARIOS)
    parser.add_argument("--algos", nargs="+", default=["PPO", "A2C", "DQN", "QLearning"])
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--ablation-env", default=None)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--skip-tune", action="store_true", help="reuse results/v2/tuned_baselines.json")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    os.makedirs(LOG_DIR, exist_ok=True)

    # Slowest algorithms first so the pool stays busy
    jobs = [([sys.executable, "-W", "ignore", "train_v2.py", "--algo", a, "--env", e, "--seed", str(s)],
             f"{a}_{e}_seed{s}.log")
            for a, e, s in itertools.product(args.algos, args.envs, args.seeds)]
    if args.ablation_env:
        for s in (args.eval_seeds or args.seeds):
            for flags, tag in [(["--no-cost_aware"], "noCost"), (["--no-anti_flapping"], "noFlap"),
                               (["--no-cost_aware", "--no-anti_flapping"], "noCost_noFlap")]:
                jobs.insert(0, ([sys.executable, "-W", "ignore", "train_v2.py", "--algo", "PPO",
                              "--env", args.ablation_env, "--seed", str(s), *flags],
                             f"PPO_{args.ablation_env}_seed{s}_{tag}.log"))
    with ThreadPoolExecutor(args.jobs) as pool:
        codes = list(pool.map(lambda j: run(*j), jobs))
    failed = sum(c != 0 for c in codes)
    print(f"{len(jobs) - failed}/{len(jobs)} training runs succeeded", flush=True)

    if not args.skip_eval:
        if not args.skip_tune:
            # Rule baselines get the same validation episodes as the RL checkpoints
            subprocess.call([sys.executable, "-W", "ignore", "tune_baselines.py", "--jobs", str(args.jobs)])
        cmd = [sys.executable, "-W", "ignore", "evaluate_v2.py", "--seeds", *map(str, args.eval_seeds or args.seeds),
               "--envs", *args.envs]
        if args.ablation_env:
            cmd += ["--ablation-env", args.ablation_env]
        subprocess.call(cmd)


if __name__ == "__main__":
    main()
