import collections
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from configs import CloudConfig
from environment import trace as trace_data

RESTART, SCALE_UP, SCALE_DOWN, CLEAR_CACHE, HOLD = range(5)
ACTION_NAMES = ["Restart", "Scale Up", "Scale Down", "Clear Cache", "Hold"]

# Observation layout (all roughly in [0, 1]; values above 1 mean overload)
(OBS_REPLICAS, OBS_PENDING, OBS_CPU, OBS_MEM, OBS_LATENCY, OBS_ERR, OBS_LOAD, OBS_TREND,
 OBS_SINCE_SCALE, OBS_LAST_SCALE_DIR, OBS_SINCE_RESTART) = range(11)
OBS_DIM = 11


class CloudSelfHealingEnvV2(gym.Env):
    """
    Replicated service under a time-varying load with injected faults.

    cpu = load / serving replicas, latency follows an M/M/1-style curve in cpu,
    each replica costs reward every step, and the reward is 1 when the SLA
    (latency and error rate) is met. Each fault has one intended remedy:
    memory leak -> restart, error burst -> clear cache, traffic surge -> scale up.
    """
    metadata = {"render_modes": []}

    def __init__(self, config: CloudConfig = CloudConfig()):
        super().__init__()
        self.config = config
        if config.trace:
            self.series = trace_data.load_series(config.trace)
            lo, hi = trace_data.split_bounds(len(self.series), config.trace_split)
            # An episode (max_steps + 1 windows) must fit inside its split
            self.trace_starts = (lo, hi - config.max_steps - 1)
            assert self.trace_starts[1] > lo, "trace split shorter than an episode"
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(0.0, 2.0, shape=(OBS_DIM,), dtype=np.float32)

    # ---- load and metrics -------------------------------------------------

    def _next_load(self):
        c = self.config
        self.ou += -c.ou_theta * self.ou + c.noise * self.np_random.normal()
        if c.trace:
            level = self.series[self.trace_start + self.t]
        else:
            level = 1.0 + c.daily_amp * math.sin(2 * math.pi * self.t / c.daily_period + self.phase)
        return max(0.05, c.base_load * level * (1.0 + self.ou) * self.surge_mult)

    def _update_metrics(self, capacity):
        cold = self.config.cold_cache_load if self.cold_cache_left > 0 else 1.0
        self.cpu = self.load * cold / max(capacity, 0.5)
        self.latency = 1.0 / (1.0 - min(self.cpu, 0.97))
        self.mem = min(1.2, 0.25 + 0.2 * min(self.cpu, 1.0) + self.leak)
        self.err = min(1.0, 0.002 + 0.5 * max(0.0, self.cpu - 0.9) + self.err_burst)

    def _obs(self):
        c = self.config
        trend = (self.load - self.prev_load) / max(self.prev_load, 0.1)
        return np.array([
            self.replicas / c.max_replicas,
            len(self.pending) / c.max_replicas,
            min(self.cpu, 2.0) / 2.0,
            self.mem / 1.2,
            min(self.latency / c.sla_latency, 2.0) / 2.0,
            self.err,
            min(self.load / c.max_replicas, 2.0),
            float(np.clip(trend, -1.0, 1.0)) / 2.0 + 0.5,
            # Flapping state, so the flapping penalty is Markov
            min(self.t - self.last_scale_t, c.flap_window) / c.flap_window,
            {None: 0.5, SCALE_UP: 1.0, SCALE_DOWN: 0.0}[self.last_scale_dir],
            min(self.t - self.last_restart_t, c.flap_window) / c.flap_window,
        ], dtype=np.float32)

    # ---- gym API ------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        c = self.config
        self.t = 0
        self.phase = self.np_random.uniform(0, 2 * math.pi)
        if c.trace:
            self.trace_start = int(self.np_random.integers(*self.trace_starts))
        self.ou = 0.0
        self.surge_left, self.surge_peak, self.surge_mult = 0, 1.0, 1.0
        self.leaking, self.leak, self.err_burst = False, 0.0, 0.0
        self.cold_cache_left = 0
        self.load = self._next_load()
        self.prev_load = self.load
        self.replicas = int(min(c.max_replicas, max(1, math.ceil(self.load / c.init_target_cpu))))
        self.pending = collections.deque()
        self.overload_steps = 0
        self.last_scale_dir, self.last_scale_t = None, -c.flap_window
        self.last_restart_t = -c.flap_window
        self.crashes = 0
        self.flapping_incidents = 0
        self._update_metrics(self.replicas)
        return self._obs(), {}

    def step(self, action):
        a = int(action)
        scale = a if a in (SCALE_UP, SCALE_DOWN) else HOLD
        heal = a if a in (RESTART, CLEAR_CACHE) else HOLD
        return self.step_joint(scale, heal)

    def step_joint(self, scale_action, heal_action):
        """One step with a scaling action (SCALE_UP / SCALE_DOWN / HOLD) and a healing action
        (RESTART / CLEAR_CACHE / HOLD) taken together. step() is the single-action case."""
        c = self.config
        self.t += 1

        # Healing action
        action_cost, lost_capacity = 0.0, 0.0
        if heal_action == RESTART:
            # Rolling restart: one replica out of rotation for this step
            self.leaking, self.leak, self.err_burst = False, 0.0, 0.0
            lost_capacity, action_cost = 1.0, c.c_restart
        elif heal_action == CLEAR_CACHE:
            # Fixes a bad-cache error burst but not a heap leak; the cache starts cold
            self.err_burst = 0.0
            self.cold_cache_left = c.cold_cache_steps + 1
            action_cost = c.c_cache
        # Scaling action
        if scale_action == SCALE_UP:
            if self.replicas + len(self.pending) < c.max_replicas:
                self.pending.append(self.t + c.provision_delay)
            action_cost += c.c_scale
        elif scale_action == SCALE_DOWN:
            if self.replicas > 1:
                self.replicas -= 1
            action_cost += c.c_scale
        while self.pending and self.pending[0] <= self.t:
            self.pending.popleft()
            self.replicas += 1

        # Flapping: scaling against the previous direction, or a second restart, within the window
        flaps = 0
        if scale_action in (SCALE_UP, SCALE_DOWN):
            if (self.last_scale_dir is not None and self.last_scale_dir != scale_action
                    and self.t - self.last_scale_t < c.flap_window):
                flaps += 1
            self.last_scale_dir, self.last_scale_t = scale_action, self.t
        if heal_action == RESTART:
            flaps += int(self.t - self.last_restart_t < c.flap_window)
            self.last_restart_t = self.t
        self.flapping_incidents += flaps
        flap = flaps > 0

        # Faults
        if not self.leaking and self.np_random.random() < c.p_leak:
            self.leaking = True
        if self.leaking:
            self.leak += c.leak_rate
        if self.err_burst == 0.0 and self.np_random.random() < c.p_err:
            self.err_burst = self.np_random.uniform(0.1, 0.3)
        if self.surge_left == 0 and self.np_random.random() < c.p_surge:
            self.surge_left = int(self.np_random.integers(c.surge_len[0], c.surge_len[1] + 1))
            self.surge_peak = self.np_random.uniform(c.surge_min, c.surge_max)
        if self.surge_left > 0:
            self.surge_left -= 1
            self.surge_mult = min(self.surge_peak, self.surge_mult + (self.surge_peak - 1.0) / c.surge_ramp)
            if self.surge_left == 0:
                self.surge_mult = 1.0

        # Dynamics
        self.prev_load, self.load = self.load, self._next_load()
        self.cold_cache_left = max(0, self.cold_cache_left - 1)
        self._update_metrics(self.replicas - lost_capacity)
        self.overload_steps = self.overload_steps + 1 if self.cpu > c.crash_cpu else 0
        crash = self.mem >= 1.0 or self.overload_steps >= c.crash_overload_steps
        sla_ok = self.latency <= c.sla_latency and self.err <= c.sla_err

        # Reward
        replica_cost = c.c_rep * self.replicas / c.max_replicas
        reward = float(sla_ok) - replica_cost
        if c.cost_aware:
            reward -= action_cost
        if c.anti_flapping:
            reward -= c.c_flap * flaps
        if crash:
            self.crashes += 1
            reward -= c.crash_penalty

        info = {
            "sla_ok": sla_ok,
            "replicas": self.replicas,
            "replica_cost": replica_cost,
            "action_cost": action_cost,
            "crash": crash,
            "flap": flap,
            "cpu": self.cpu,
            "fault_leak": self.leaking,
            "fault_err": self.err_burst > 0.0,
            "fault_surge": self.surge_left > 0,
        }
        return self._obs(), reward, crash, self.t >= c.max_steps, info
