"""
Two-agent version of the v2 environment for RLlib.

  scaler: 0 = hold, 1 = scale up, 2 = scale down
  healer: 0 = hold, 1 = restart, 2 = clear cache

Both agents act every step on the same service and receive the same (team) reward.
Each observation is the 11 shared service metrics, a one-hot agent id (so one shared
policy can serve both agents) and, when `communicate` is on, the other agent's previous
action as a one-hot message (zeros otherwise, so every variant has the same input size).
"""
import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from configs import SCENARIOS, for_split
from environment.cloud_env_v2 import (
    CloudSelfHealingEnvV2, OBS_DIM, HOLD, SCALE_UP, SCALE_DOWN, RESTART, CLEAR_CACHE,
)

AGENTS = ["scaler", "healer"]
SCALER_ACTIONS = [HOLD, SCALE_UP, SCALE_DOWN]
HEALER_ACTIONS = [HOLD, RESTART, CLEAR_CACHE]
MA_OBS_DIM = OBS_DIM + len(AGENTS) + 3


def make_config(env_config):
    config = SCENARIOS[env_config.get("scenario", "Mixed")]
    return for_split(config, env_config.get("split", "train"))


class MultiAgentCloudEnv(MultiAgentEnv):
    def __init__(self, env_config=None):
        super().__init__()
        env_config = env_config or {}
        self.env = CloudSelfHealingEnvV2(make_config(env_config))
        self.communicate = env_config.get("communicate", True)
        self.reward_scale = env_config.get("reward_scale", 1.0)
        self.agents = self.possible_agents = list(AGENTS)
        obs_space = spaces.Box(0.0, 2.0, shape=(MA_OBS_DIM,), dtype=np.float32)
        self.observation_spaces = {a: obs_space for a in AGENTS}
        self.action_spaces = {a: spaces.Discrete(3) for a in AGENTS}
        self.last_actions = {a: 0 for a in AGENTS}

    def _obs(self, base):
        out = {}
        for i, agent in enumerate(AGENTS):
            agent_id = np.zeros(len(AGENTS), dtype=np.float32)
            agent_id[i] = 1.0
            message = np.zeros(3, dtype=np.float32)
            if self.communicate:
                message[self.last_actions[AGENTS[1 - i]]] = 1.0
            out[agent] = np.concatenate([base, agent_id, message]).astype(np.float32)
        return out

    def reset(self, *, seed=None, options=None):
        base, info = self.env.reset(seed=seed)
        self.last_actions = {a: 0 for a in AGENTS}
        return self._obs(base), {a: {} for a in AGENTS}

    def step(self, action_dict):
        scale = int(action_dict.get("scaler", 0))
        heal = int(action_dict.get("healer", 0))
        self.last_actions = {"scaler": scale, "healer": heal}
        base, reward, terminated, truncated, info = self.env.step_joint(SCALER_ACTIONS[scale], HEALER_ACTIONS[heal])
        r = reward * self.reward_scale
        done = {"__all__": terminated}
        trunc = {"__all__": truncated}
        info = {**info, "raw_reward": reward}
        return self._obs(base), {a: r for a in AGENTS}, done, trunc, {a: info for a in AGENTS}
