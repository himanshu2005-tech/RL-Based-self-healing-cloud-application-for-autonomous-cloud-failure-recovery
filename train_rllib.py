"""
Train with RLlib PPO: the single-agent v2 env, or the two-agent (scaler + healer) env.

    python train_rllib.py --mode single --env Mixed --seed 1
    python train_rllib.py --mode multi --policy separate --comm --env TraceFaults --seed 1
    python train_rllib.py --mode multi --policy shared --no-comm --env TraceFaults --seed 1

Like train_v2.py, the checkpoint with the best mean reward on the validation episodes is
kept. Outputs: models/v2/rllib/<prefix>/<module id>/ (RLModule checkpoints),
results/v2/<prefix>_train.csv and results/v2/<prefix>_val.csv.
"""
import argparse
import os
import random
import shutil

import numpy as np
import pandas as pd
import torch

from configs import MODEL_DIR, RESULT_DIR, SCENARIOS, VAL_SEEDS, for_split
from environment.cloud_env_v2 import CloudSelfHealingEnvV2
from environment.multi_agent_env import AGENTS, MultiAgentCloudEnv

RLLIB_DIR = os.path.join(MODEL_DIR, "rllib")
REWARD_SCALE = 0.1   # training rewards only; validation and evaluation use raw rewards
RAY_TMP = os.environ.get("RAY_TMPDIR", r"E:\rt")


def run_prefix(mode, env_name, seed, policy="separate", comm=True):
    if mode == "single":
        return f"RLlibPPO_{env_name}_seed{seed}"
    return f"MA-{policy}-{'comm' if comm else 'nocomm'}_{env_name}_seed{seed}"


class SingleAgentEnv(CloudSelfHealingEnvV2):
    """v2 env built from an RLlib env_config, with scaled training rewards."""

    def __init__(self, env_config=None):
        env_config = env_config or {}
        super().__init__(for_split(SCENARIOS[env_config["scenario"]], env_config.get("split", "train")))
        self.reward_scale = env_config.get("reward_scale", 1.0)

    def step(self, action):
        obs, r, terminated, truncated, info = super().step(action)
        return obs, r * self.reward_scale, terminated, truncated, {**info, "raw_reward": r}


def greedy(module):
    def act(obs):
        with torch.no_grad():
            out = module.forward_inference({"obs": torch.as_tensor(obs[None], dtype=torch.float32)})
        return int(torch.argmax(out["action_dist_inputs"][0]))
    return act


def ma_policy_fn(modules, policy):
    """Joint action for both agents from per-agent observations."""
    acts = {a: greedy(modules["shared" if policy == "shared" else a]) for a in AGENTS}
    return lambda obs: {a: acts[a](obs[a]) for a in AGENTS}


def run_episode_ma(env, act, seed):
    obs, _ = env.reset(seed=seed)
    total, steps, sla, replicas = 0.0, 0, 0, 0.0
    while True:
        obs, _, done, trunc, info = env.step(act(obs))
        i = info["scaler"]
        total += i["raw_reward"]
        steps += 1
        sla += int(i["sla_ok"])
        replicas += i["replicas"]
        if done["__all__"] or trunc["__all__"]:
            return total, steps, sla, replicas, env.env


def validate(args, modules):
    if args.mode == "single":
        env = CloudSelfHealingEnvV2(for_split(SCENARIOS[args.env], "val"))
        act = greedy(modules["default_policy"])
        totals = []
        for s in VAL_SEEDS:
            obs, _ = env.reset(seed=s)
            done, total = False, 0.0
            while not done:
                obs, r, te, tr, _ = env.step(act(obs))
                total += r
                done = te or tr
            totals.append(total)
        return float(np.mean(totals))
    env = MultiAgentCloudEnv({"scenario": args.env, "split": "val", "communicate": args.comm})
    act = ma_policy_fn(modules, args.policy)
    return float(np.mean([run_episode_ma(env, act, s)[0] for s in VAL_SEEDS]))


def build(args):
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

    env_config = {"scenario": args.env, "split": "train", "reward_scale": REWARD_SCALE}
    config = PPOConfig()
    if args.mode == "single":
        config = config.environment(SingleAgentEnv, env_config=env_config)
    else:
        env_config["communicate"] = args.comm
        config = config.environment(MultiAgentCloudEnv, env_config=env_config)
        if args.policy == "shared":
            config = config.multi_agent(policies={"shared"}, policy_mapping_fn=lambda agent_id, *a, **k: "shared")
        else:
            config = config.multi_agent(policies=set(AGENTS), policy_mapping_fn=lambda agent_id, *a, **k: agent_id)
    config = (config
              .env_runners(num_env_runners=0, num_envs_per_env_runner=4)
              .training(lr=3e-4, gamma=0.99, lambda_=0.95, train_batch_size_per_learner=4096,
                        minibatch_size=256, num_epochs=10, entropy_coeff=0.01, vf_clip_param=100.0)
              .rl_module(model_config=DefaultModelConfig(fcnet_hiddens=[64, 64]))
              .debugging(seed=args.seed))
    return config.build_algo()


def module_ids(args):
    if args.mode == "single":
        return ["default_policy"]
    return ["shared"] if args.policy == "shared" else list(AGENTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "multi"], required=True)
    parser.add_argument("--policy", choices=["separate", "shared"], default="separate")
    parser.add_argument("--comm", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--env", choices=list(SCENARIOS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--eval-every", type=int, default=20_000)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    prefix = run_prefix(args.mode, args.env, args.seed, args.policy, args.comm)
    out_dir = os.path.abspath(os.path.join(RLLIB_DIR, prefix))
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"Training {prefix}", flush=True)

    import ray
    from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME, EPISODE_RETURN_MEAN
    if os.environ.get("RAY_ADDRESS"):
        # Join a shared head node (run_trace_experiments.py starts one); several runs at once
        # each starting their own local cluster overloads a laptop
        ray.init(address=os.environ["RAY_ADDRESS"], log_to_driver=False, configure_logging=False)
    else:
        os.makedirs(RAY_TMP, exist_ok=True)
        ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False, _temp_dir=RAY_TMP,
                 ignore_reinit_error=True, configure_logging=False)
    algo = build(args)

    train_rows, val_rows = [], []
    best, next_eval, steps = -np.inf, args.eval_every, 0
    while steps < args.timesteps:
        result = algo.train()
        env_results = result[ENV_RUNNER_RESULTS]
        steps = int(env_results[NUM_ENV_STEPS_SAMPLED_LIFETIME])
        ret = env_results.get(EPISODE_RETURN_MEAN, np.nan)
        # Multi-agent returns are summed over both agents; report the team return once
        if args.mode == "multi":
            ret = ret / len(AGENTS)
        train_rows.append({"timestep": steps, "episode_return_mean": ret / REWARD_SCALE})
        if steps >= next_eval:
            next_eval += args.eval_every
            modules = {m: algo.get_module(m) for m in module_ids(args)}
            score = validate(args, modules)
            val_rows.append({"timestep": steps, "val_reward": score})
            print(f"  {steps:>7d} steps  val {score:7.1f}", flush=True)
            if score > best:
                best = score
                shutil.rmtree(out_dir, ignore_errors=True)
                for m, module in modules.items():
                    module.save_to_path(os.path.join(out_dir, m))

    pd.DataFrame(train_rows).to_csv(os.path.join(RESULT_DIR, f"{prefix}_train.csv"), index=False)
    pd.DataFrame(val_rows).to_csv(os.path.join(RESULT_DIR, f"{prefix}_val.csv"), index=False)
    algo.stop()
    ray.shutdown()
    print(f"Done {prefix}: best validation reward {best:.1f}", flush=True)


if __name__ == "__main__":
    main()
