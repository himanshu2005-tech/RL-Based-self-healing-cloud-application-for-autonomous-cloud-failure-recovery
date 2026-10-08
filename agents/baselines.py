import math

from environment.cloud_env_v2 import (
    RESTART, SCALE_UP, SCALE_DOWN, CLEAR_CACHE, HOLD,
    OBS_REPLICAS, OBS_PENDING, OBS_CPU, OBS_MEM, OBS_ERR,
)


class HoldAgent:
    """Never acts: the service keeps its initial size."""

    def reset(self):
        pass

    def act(self, obs):
        return HOLD


class HPAAgent:
    """
    Kubernetes Horizontal Pod Autoscaler rule, reading only the observation:
    desired = ceil(replicas * cpu / target_cpu), scale up one replica at a time,
    scale down only when every recommendation in the stabilisation window is lower.
    With heal=True, also restart on high memory and clear the cache on errors that
    are not explained by overload.
    """

    def __init__(self, max_replicas=12, target_cpu=0.6, stabilization=5, heal=False,
                 mem_threshold=0.8, err_threshold=0.05, overload_cpu=0.9):
        self.max_replicas = max_replicas
        self.target_cpu = target_cpu
        self.stabilization = stabilization
        self.heal = heal
        self.mem_threshold = mem_threshold
        self.err_threshold = err_threshold
        self.overload_cpu = overload_cpu
        self.history = []

    def reset(self):
        self.history = []

    def act(self, obs):
        replicas = round(obs[OBS_REPLICAS] * self.max_replicas)
        pending = round(obs[OBS_PENDING] * self.max_replicas)
        cpu = obs[OBS_CPU] * 2.0
        mem = obs[OBS_MEM] * 1.2
        err = obs[OBS_ERR]

        if self.heal:
            if mem > self.mem_threshold:
                return RESTART
            if err > self.err_threshold and cpu < self.overload_cpu:
                return CLEAR_CACHE

        desired = max(1, min(self.max_replicas, math.ceil(replicas * cpu / self.target_cpu)))
        self.history = (self.history + [desired])[-self.stabilization:]
        if desired > replicas + pending:
            return SCALE_UP
        if max(self.history) < replicas + pending:
            return SCALE_DOWN
        return HOLD


class ThresholdAgent:
    """
    Classic threshold autoscaler with healing rules, reading only the observation.
    Scale up while the cpu projected over replicas + pending exceeds `up`; scale down
    when cpu < `down`, at least `cooldown` steps after the last scaling action, and only
    if one fewer replica would stay below `up`. Restart above `mem` memory; clear the
    cache above `err` errors when the errors are not explained by overload.
    """

    def __init__(self, max_replicas=12, up=0.7, down=0.35, cooldown=3, mem=0.8, err=0.05, overload_cpu=0.9):
        self.max_replicas = max_replicas
        self.up, self.down, self.cooldown = up, down, cooldown
        self.mem, self.err, self.overload_cpu = mem, err, overload_cpu
        self.since_scale = cooldown

    def reset(self):
        self.since_scale = self.cooldown

    def act(self, obs):
        replicas = round(obs[OBS_REPLICAS] * self.max_replicas)
        pending = round(obs[OBS_PENDING] * self.max_replicas)
        cpu = obs[OBS_CPU] * 2.0
        self.since_scale += 1

        if obs[OBS_MEM] * 1.2 > self.mem:
            return RESTART
        if obs[OBS_ERR] > self.err and cpu < self.overload_cpu:
            return CLEAR_CACHE
        if cpu * replicas / (replicas + pending) > self.up and replicas + pending < self.max_replicas:
            self.since_scale = 0
            return SCALE_UP
        if (cpu < self.down and replicas > 1 and pending == 0 and self.since_scale >= self.cooldown
                and cpu * replicas / (replicas - 1) < self.up):
            self.since_scale = 0
            return SCALE_DOWN
        return HOLD
