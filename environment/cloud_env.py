import gymnasium as gym
from gymnasium import spaces
import numpy as np
import collections

from configs import EnvConfig

class CloudSelfHealingEnv(gym.Env):
    """
    Custom Environment that follows gym interface for self-healing cloud application.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: EnvConfig = EnvConfig()):
        super().__init__()
        self.config = config
        
        # Actions: 0=restart, 1=scale_up, 2=scale_down, 3=clear_cache, 4=do_nothing
        self.action_space = spaces.Discrete(5)
        
        # State: [cpu_usage, ram_usage, response_time, error_rate, request_load]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 10.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action history for anti-flapping
        self.action_history = collections.deque(maxlen=self.config.flapping_window)
        self.flapping_incidents = 0
        self.crashes = 0
        
        # Initialize state
        self.state = None
        self.current_step = 0
        self.max_steps = 200

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.action_history.clear()
        self.flapping_incidents = 0
        self.crashes = 0
        
        # Start near baseline
        cpu = np.clip(self.np_random.normal(self.config.cpu_baseline_mean, self.config.cpu_baseline_std), 0.0, 1.0)
        ram = np.clip(self.np_random.normal(self.config.ram_baseline_mean, self.config.ram_baseline_std), 0.0, 1.0)
        resp_time = np.clip(self.np_random.normal(0.5, 0.1), 0.0, 10.0)
        err_rate = np.clip(self.np_random.normal(0.01, 0.005), 0.0, 1.0)
        req_load = np.clip(self.np_random.normal(0.3, 0.1), 0.0, 1.0)
        
        self.state = np.array([cpu, ram, resp_time, err_rate, req_load], dtype=np.float32)
        info = {}
        return self.state, info

    def step(self, action):
        action = int(action)
        self.current_step += 1
        cpu, ram, resp_time, err_rate, req_load = self.state
        
        # Action Costs & Effects
        action_cost = 0.0
        
        if action == 0:  # Restart
            cpu = np.clip(self.np_random.normal(self.config.cpu_baseline_mean, self.config.cpu_baseline_std), 0.0, 1.0)
            ram = np.clip(self.np_random.normal(self.config.ram_baseline_mean, self.config.ram_baseline_std), 0.0, 1.0)
            err_rate = 0.0
            resp_time = 5.0  # Spikes due to restart
            action_cost = self.config.cost_restart
            
        elif action == 1:  # Scale Up
            cpu = np.clip(cpu * 0.6, 0.0, 1.0)
            ram = np.clip(ram * 0.6, 0.0, 1.0)
            resp_time = np.clip(resp_time * 0.8, 0.0, 10.0)
            action_cost = self.config.cost_scale_up
            
        elif action == 2:  # Scale Down
            cpu = np.clip(cpu * 1.5, 0.0, 1.0)
            ram = np.clip(ram * 1.5, 0.0, 1.0)
            action_cost = self.config.cost_scale_down
            
        elif action == 3:  # Clear Cache
            resp_time = np.clip(resp_time * 0.9, 0.0, 10.0)
            err_rate = np.clip(err_rate * 0.8, 0.0, 1.0)
            action_cost = self.config.cost_clear_cache
            
        elif action == 4:  # Do Nothing
            action_cost = self.config.cost_do_nothing
            
        # Natural Drift & Spikes
        if self.np_random.random() < self.config.spike_prob:
            req_load = np.clip(req_load + 0.4, 0.0, 1.0)
        else:
            req_load = np.clip(req_load + self.np_random.normal(0.0, 0.05), 0.0, 1.0)
            
        # CPU/RAM drift towards load
        cpu = np.clip(cpu + 0.1 * (req_load - cpu) + self.np_random.normal(0.0, 0.02), 0.0, 1.0)
        ram = np.clip(ram + 0.1 * (req_load - ram) + self.np_random.normal(0.0, 0.02), 0.0, 1.0)
        
        # Response time correlates with CPU
        resp_time = np.clip(resp_time * 0.5 + 5.0 * cpu + self.np_random.normal(0.0, 0.2), 0.0, 10.0)
        
        # Errors happen when CPU/RAM are stressed
        if cpu > 0.85 or ram > 0.85:
            err_rate = np.clip(err_rate + self.np_random.normal(0.1, 0.05), 0.0, 1.0)
        else:
            err_rate = np.clip(err_rate - 0.05, 0.0, 1.0)
            
        # Check for crash
        crash = False
        crash_penalty = 0.0
        if cpu > 0.98 or err_rate > 0.50:
            if self.np_random.random() < self.config.failure_frequency * 10:  # Higher chance if stressed
                crash = True
                self.crashes += 1
                crash_penalty = self.config.w3
                # Auto-recover somewhat from crash
                cpu, ram, err_rate, resp_time = 0.5, 0.5, 0.0, 8.0

        self.state = np.array([cpu, ram, resp_time, err_rate, req_load], dtype=np.float32)

        # Reward Calculation
        health_score = 1.0 - (cpu + ram + (resp_time/10.0) + err_rate) / 4.0
        
        reward = self.config.w1 * health_score
        
        if self.config.cost_aware:
            reward -= self.config.w2 * action_cost
            
        reward -= crash_penalty

        # Anti-Flapping
        flapping_penalty_applied = 0.0
        disruptive_actions = {0, 1, 2}
        if action in disruptive_actions:
            self.action_history.append(action)
            if self.config.anti_flapping and len(self.action_history) == self.config.flapping_window:
                # Flapping: same action repeated or rapid alternation
                # Actually, any disruptive actions rapidly filling the window is considered flapping for this project
                reward -= self.config.flapping_penalty
                flapping_penalty_applied = self.config.flapping_penalty
                self.flapping_incidents += 1
                # Clear history so we don't penalize every single step afterwards, just when a burst of them happen
                self.action_history.clear()

        terminated = False
        truncated = self.current_step >= self.max_steps
        
        info = {
            "health_score": health_score,
            "action_cost": action_cost,
            "crash": crash,
            "flapping_penalty": flapping_penalty_applied
        }
        
        return self.state, reward, terminated, truncated, info