import dataclasses

import numpy as np
import pandas as pd
import pytest

from configs import CloudConfig, SCENARIOS
from environment import trace as trace_data
from environment.cloud_env_v2 import CloudSelfHealingEnvV2, HOLD, RESTART, SCALE_UP, CLEAR_CACHE


@pytest.fixture
def fake_trace(tmp_path, monkeypatch):
    # 2000 windows whose value encodes the window index, so the split used is visible
    pd.DataFrame({"window": range(2000), "load": np.arange(2000) + 1.0}).to_csv(tmp_path / "fake.csv", index=False)
    monkeypatch.setattr(trace_data, "TRACE_DIR", str(tmp_path))
    trace_data.load_series.cache_clear()
    yield "fake"
    trace_data.load_series.cache_clear()


def windows_used(config, seeds):
    env = CloudSelfHealingEnvV2(config)
    used = set()
    for s in seeds:
        env.reset(seed=s)
        used.update(range(env.trace_start, env.trace_start + config.max_steps + 1))
    return used


def test_trace_splits_do_not_overlap(fake_trace):
    base = CloudConfig(trace=fake_trace, noise=0.0)
    train = windows_used(base, range(200))
    val = windows_used(dataclasses.replace(base, trace_split="val"), range(200))
    test = windows_used(dataclasses.replace(base, trace_split="test"), range(200))
    assert not train & val and not train & test and not val & test
    assert max(train) < min(val) and max(val) < min(test)


def test_trace_drives_the_load(fake_trace):
    cfg = CloudConfig(trace=fake_trace, noise=0.0, base_load=1.0)
    env = CloudSelfHealingEnvV2(cfg)
    env.reset(seed=0)
    series = trace_data.load_series(fake_trace)
    for _ in range(5):
        env.step(HOLD)
        assert env.load == pytest.approx(series[env.trace_start + env.t])


def test_series_normalised_on_training_split(fake_trace):
    x = trace_data.load_series(fake_trace)
    lo, hi = trace_data.split_bounds(len(x), "train")
    assert x[lo:hi].mean() == pytest.approx(1.0)


def test_single_action_equals_joint_step():
    a, b = CloudSelfHealingEnvV2(SCENARIOS["Mixed"]), CloudSelfHealingEnvV2(SCENARIOS["Mixed"])
    a.reset(seed=3)
    b.reset(seed=3)
    for act in [SCALE_UP, HOLD, RESTART, CLEAR_CACHE, HOLD, SCALE_UP]:
        ra = a.step(act)
        scale = act if act == SCALE_UP else HOLD
        heal = act if act in (RESTART, CLEAR_CACHE) else HOLD
        rb = b.step_joint(scale, heal)
        assert ra[1] == rb[1] and np.array_equal(ra[0], rb[0])


def test_joint_step_applies_both_actions():
    env = CloudSelfHealingEnvV2(dataclasses.replace(SCENARIOS["Low"], p_leak=1.0))
    env.reset(seed=0)
    env.step(HOLD)
    assert env.leak > 0
    env.step_joint(SCALE_UP, RESTART)
    assert env.leak == pytest.approx(env.config.leak_rate) and len(env.pending) == 1


def test_multi_agent_messages():
    from environment.multi_agent_env import MultiAgentCloudEnv, OBS_DIM
    on = MultiAgentCloudEnv({"scenario": "Mixed", "communicate": True})
    off = MultiAgentCloudEnv({"scenario": "Mixed", "communicate": False})
    for env in (on, off):
        env.reset(seed=1)
    o_on, r_on, *_ = on.step({"scaler": 1, "healer": 2})
    o_off, r_off, *_ = off.step({"scaler": 1, "healer": 2})
    # The healer sees the scaler's last action (scale up = 1) and vice versa; silent without comm
    assert list(o_on["healer"][-3:]) == [0, 1, 0] and list(o_on["scaler"][-3:]) == [0, 0, 1]
    assert not o_off["healer"][-3:].any()
    # Same dynamics and team reward with or without the message channel
    assert r_on == r_off and r_on["scaler"] == r_on["healer"]
    assert np.array_equal(o_on["scaler"][:OBS_DIM], o_off["scaler"][:OBS_DIM])
    # Agent ids let one shared policy tell the agents apart
    assert list(o_on["scaler"][OBS_DIM:OBS_DIM + 2]) == [1, 0] and list(o_on["healer"][OBS_DIM:OBS_DIM + 2]) == [0, 1]
