import dataclasses

import numpy as np
import pytest

from agents.baselines import HPAAgent
from configs import SCENARIOS, CloudConfig
from environment.cloud_env_v2 import (
    CloudSelfHealingEnvV2, RESTART, SCALE_UP, SCALE_DOWN, CLEAR_CACHE, HOLD,
)


def rollout(env, actions, seed=0):
    env.reset(seed=seed)
    infos = []
    for a in actions:
        _, _, terminated, truncated, info = env.step(a)
        infos.append(info)
        if terminated or truncated:
            break
    return infos


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_observations_stay_in_space(name):
    env = CloudSelfHealingEnvV2(SCENARIOS[name])
    obs, _ = env.reset(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(200):
        assert env.observation_space.contains(obs)
        obs, _, terminated, truncated, _ = env.step(int(rng.integers(5)))
        if terminated or truncated:
            obs, _ = env.reset()


def test_same_seed_same_episode():
    a = rollout(CloudSelfHealingEnvV2(SCENARIOS["Mixed"]), [HOLD] * 50, seed=7)
    b = rollout(CloudSelfHealingEnvV2(SCENARIOS["Mixed"]), [HOLD] * 50, seed=7)
    assert [i["cpu"] for i in a] == [i["cpu"] for i in b]


def test_low_and_high_differ():
    def mean_load(name):
        env = CloudSelfHealingEnvV2(SCENARIOS[name])
        env.reset(seed=0)
        return np.mean([env.load for _ in range(100) if env.step(HOLD)])
    assert mean_load("High") > 2 * mean_load("Low")


def test_load_is_mean_reverting():
    env = CloudSelfHealingEnvV2(dataclasses.replace(CloudConfig(), daily_amp=0.0))
    loads = []
    for s in range(20):
        env.reset(seed=s)
        for _ in range(200):
            env.step(HOLD)
            loads.append(env.load)
    assert abs(np.mean(loads) / CloudConfig().base_load - 1) < 0.1


def test_scale_up_arrives_after_provision_delay():
    env = CloudSelfHealingEnvV2(SCENARIOS["Low"])
    env.reset(seed=0)
    n = env.replicas
    env.step(SCALE_UP)
    assert env.replicas == n and len(env.pending) == 1
    env.step(HOLD)
    assert env.replicas == n
    env.step(HOLD)
    assert env.replicas == n + 1


def test_more_replicas_lower_cpu():
    env = CloudSelfHealingEnvV2(SCENARIOS["Low"])
    env.reset(seed=0)
    env.load = 3.0
    env._update_metrics(3)
    cpu3 = env.cpu
    env._update_metrics(6)
    assert env.cpu == pytest.approx(cpu3 / 2)


def test_memory_leak_needs_restart():
    cfg = dataclasses.replace(SCENARIOS["Low"], p_leak=1.0)
    # Clearing the cache every step does not stop a leak
    infos = rollout(CloudSelfHealingEnvV2(cfg), [CLEAR_CACHE] * 200)
    assert infos[-1]["crash"]
    # Restarting before memory fills avoids the OOM
    actions = [RESTART if t % 15 == 14 else HOLD for t in range(200)]
    infos = rollout(CloudSelfHealingEnvV2(cfg), actions)
    assert not any(i["crash"] for i in infos)


def test_clear_cache_fixes_error_burst():
    env = CloudSelfHealingEnvV2(dataclasses.replace(SCENARIOS["Low"], p_err=1.0))
    env.reset(seed=0)
    env.step(HOLD)
    assert env.err_burst > 0
    env.step(CLEAR_CACHE)
    # A new burst may start in the same step (p_err=1), so check the fix via a no-new-burst config
    env.config = dataclasses.replace(env.config, p_err=0.0)
    env.step(CLEAR_CACHE)
    assert env.err_burst == 0


def test_crash_terminates_with_penalty():
    cfg = dataclasses.replace(SCENARIOS["Low"], p_leak=1.0)
    env = CloudSelfHealingEnvV2(cfg)
    env.reset(seed=0)
    for _ in range(200):
        _, r, terminated, _, info = env.step(HOLD)
        if terminated:
            break
    assert terminated and info["crash"] and r < -cfg.crash_penalty + 1


def test_flapping_is_reversal_not_repetition():
    env = CloudSelfHealingEnvV2(SCENARIOS["Low"])
    infos = rollout(env, [SCALE_UP] * 4)
    assert env.flapping_incidents == 0
    infos = rollout(env, [SCALE_UP, SCALE_DOWN])
    assert env.flapping_incidents == 1
    # A reversal outside the window is not flapping
    rollout(env, [SCALE_UP] + [HOLD] * 10 + [SCALE_DOWN])
    assert env.flapping_incidents == 0
    rollout(env, [RESTART, HOLD, RESTART])
    assert env.flapping_incidents == 1


def test_flapping_recorded_even_without_penalty():
    cfg = dataclasses.replace(SCENARIOS["Low"], anti_flapping=False)
    env = CloudSelfHealingEnvV2(cfg)
    rollout(env, [SCALE_UP, SCALE_DOWN])
    assert env.flapping_incidents == 1


def test_ablation_flags_change_only_the_reward():
    base = SCENARIOS["Mixed"]
    actions = list(np.random.default_rng(0).integers(5, size=100))
    a = rollout(CloudSelfHealingEnvV2(base), actions, seed=3)
    b = rollout(CloudSelfHealingEnvV2(dataclasses.replace(base, cost_aware=False, anti_flapping=False)), actions, seed=3)
    assert [i["cpu"] for i in a] == [i["cpu"] for i in b]


def test_idle_is_not_free():
    # Over-provisioning lowers reward even when the SLA is met
    env = CloudSelfHealingEnvV2(SCENARIOS["Low"])
    env.reset(seed=0)
    _, r_small, *_ = env.step(HOLD)
    env.reset(seed=0)
    env.replicas = 12
    _, r_big, *_ = env.step(HOLD)
    assert r_big < r_small


def test_hpa_scales_with_load():
    hpa = HPAAgent()
    obs = np.zeros(11, dtype=np.float32)
    obs[0] = 4 / 12          # 4 replicas
    obs[2] = 0.9 / 2         # 90% cpu
    assert hpa.act(obs) == SCALE_UP
