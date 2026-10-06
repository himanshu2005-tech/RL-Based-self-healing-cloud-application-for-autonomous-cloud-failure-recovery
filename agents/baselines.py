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
