"""
Trace stage: Borg-trace-driven scenarios, single-agent (SB3 + RLlib) and multi-agent (RLlib).

    python run_trace_experiments.py --seeds 1 2 3 4 5

1. build the load series from the downloaded trace parts (data_analysis/build_trace_series.py)
2. tune the rule baselines on the validation split
3. train SB3 PPO / A2C / DQN / Q-learning and RLlib PPO on Trace and TraceFaults, and the
   four multi-agent variants (separate / shared policies x with / without communication)
   on TraceFaults
4. evaluate everything on the test split and run the significance analysis
"""
import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from configs import TRACE_SCENARIOS
from train_rllib import RAY_TMP

LOG_DIR = os.path.join("results", "v2", "logs")
PY = [sys.executable, "-W", "ignore"]


def run(cmd, log_name):
    with open(os.path.join(LOG_DIR, log_name), "w") as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f"{'ok' if code == 0 else 'FAILED'}  {log_name}", flush=True)
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--sb3-jobs", type=int, default=4)
    parser.add_argument("--rllib-jobs", type=int, default=2)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    os.makedirs(LOG_DIR, exist_ok=True)

    if not args.skip_build:
        subprocess.check_call(PY + ["data_analysis/build_trace_series.py"])
        subprocess.check_call(PY + ["tune_baselines.py", "--envs", *TRACE_SCENARIOS, "--jobs", "8"])

    if not args.skip_train:
        sb3 = [(PY + ["train_v2.py", "--algo", a, "--env", e, "--seed", str(s)], f"{a}_{e}_seed{s}.log")
               for a, e, s in itertools.product(["DQN", "PPO", "A2C", "QLearning"], TRACE_SCENARIOS, args.seeds)]
        rllib = [(PY + ["train_rllib.py", "--mode", "multi", "--policy", p, "--comm" if c else "--no-comm",
                        "--env", "TraceFaults", "--seed", str(s)],
                  f"MA-{p}-{'comm' if c else 'nocomm'}_TraceFaults_seed{s}.log")
                 for p, c, s in itertools.product(["separate", "shared"], [True, False], args.seeds)]
        rllib += [(PY + ["train_rllib.py", "--mode", "single", "--env", e, "--seed", str(s)], f"RLlibPPO_{e}_seed{s}.log")
                  for e, s in itertools.product(TRACE_SCENARIOS, args.seeds)]
        # One shared Ray head for all RLlib runs. num-cpus=1 keeps Ray from prestarting a worker
        # per CPU; the runs sample in their own process (num_env_runners=0) and need no workers.
        ray = os.path.join(os.path.dirname(sys.executable), "ray")
        subprocess.check_call([ray, "start", "--head", "--num-cpus=1", f"--temp-dir={RAY_TMP}",
                               "--include-dashboard=false", "--disable-usage-stats"])
        os.environ["RAY_ADDRESS"] = "auto"
        try:
            with ThreadPoolExecutor(args.sb3_jobs) as p1, ThreadPoolExecutor(args.rllib_jobs) as p2:
                futures = [p1.submit(run, *j) for j in sb3] + [p2.submit(run, *j) for j in rllib]
                failed = sum(f.result() != 0 for f in futures)
        finally:
            subprocess.call([ray, "stop", "--force"])
            os.environ.pop("RAY_ADDRESS", None)
        print(f"{len(futures) - failed}/{len(futures)} training runs succeeded", flush=True)

    algos = ["Hold", "HPA", "HPA+Heal", "Tuned HPA+Heal", "Tuned Threshold", "QLearning", "DQN", "A2C", "PPO",
             "RLlib PPO", "MA separate", "MA separate+comm", "MA shared", "MA shared+comm"]
    subprocess.call(PY + ["evaluate_v2.py", "--envs", *TRACE_SCENARIOS, "--seeds", *map(str, args.seeds),
                          "--algos", *algos, "--tag", "trace"])
    subprocess.call(PY + ["analyze_v2.py", "--tag", "trace"])


if __name__ == "__main__":
    main()
