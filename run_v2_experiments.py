"""
Train every algorithm on every v2 scenario and seed in parallel, then evaluate.

    python run_v2_experiments.py --seeds 1 2 3 --jobs 6
"""
import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from configs import SCENARIOS

LOG_DIR = os.path.join("results", "v2", "logs")


def run(cmd, log_name):
    with open(os.path.join(LOG_DIR, log_name), "w") as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f"{'ok' if code == 0 else 'FAILED'}  {log_name}", flush=True)
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--envs", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--algos", nargs="+", default=["PPO", "DQN", "QLearning"])
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    os.makedirs(LOG_DIR, exist_ok=True)

    # Slowest algorithms first so the pool stays busy
    jobs = [([sys.executable, "-W", "ignore", "train_v2.py", "--algo", a, "--env", e, "--seed", str(s)],
             f"{a}_{e}_seed{s}.log")
            for a, e, s in itertools.product(args.algos, args.envs, args.seeds)]
    with ThreadPoolExecutor(args.jobs) as pool:
        codes = list(pool.map(lambda j: run(*j), jobs))
    failed = sum(c != 0 for c in codes)
    print(f"{len(jobs) - failed}/{len(jobs)} training runs succeeded", flush=True)

    if not args.skip_eval:
        subprocess.call([sys.executable, "-W", "ignore", "evaluate_v2.py", "--seeds", *map(str, args.seeds),
                         "--envs", *args.envs])


if __name__ == "__main__":
    main()
