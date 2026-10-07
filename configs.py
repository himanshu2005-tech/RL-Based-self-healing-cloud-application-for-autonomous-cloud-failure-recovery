import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any

CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "data_analysis", "calibration_summary.json")

# Default fallback values if calibration file is missing
DEFAULT_STATS = {
    "cpu": {"mean": 0.05, "std": 0.02},
    "ram": {"mean": 0.05, "std": 0.02},
    "arrival_rate": {"lambda_steady": 10.0, "lambda_bursty": 50.0, "spike_prob": 0.05}
}

def load_calibration_stats() -> Dict[str, Any]:
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, "r") as f:
            return json.load(f)
    return DEFAULT_STATS

stats = load_calibration_stats()

@dataclass
class EnvConfig:
    # Traffic profile: "steady", "high_load", "bursty"
    traffic_profile: str = "steady"
    
    # Environment dynamics
    cpu_baseline_mean: float = stats["cpu"]["mean"]
    cpu_baseline_std: float = stats["cpu"]["std"]
    ram_baseline_mean: float = stats["ram"]["mean"]
    ram_baseline_std: float = stats["ram"]["std"]
    
    # Traffic simulation
    lambda_arrival: float = stats["arrival_rate"]["lambda_steady"]
    spike_prob: float = stats["arrival_rate"]["spike_prob"]
    failure_frequency: float = 0.01  # Probability of random failure/spike
    
    # Ablation Flags (Rigor requirements)
    cost_aware: bool = True
    anti_flapping: bool = True
    
    # Costs
    cost_restart: float = 1.0       # High downtime
    cost_scale_up: float = 0.5      # Resource $
    cost_scale_down: float = 0.1    # Minor reshuffling cost
    cost_clear_cache: float = 0.05  # Very low cost
    cost_do_nothing: float = 0.0
    
    # Rewards
    w1: float = 1.0                 # Health weight
    w2: float = 1.0                 # Action cost weight
    w3: float = 5.0                 # Crash penalty weight
    flapping_penalty: float = 2.0   # Penalty for flapping
    
    # Anti-Flapping
    flapping_window: int = 3        # Number of recent actions to track
    
    def __post_init__(self):
        # Adjust dynamics based on traffic profile
        if self.traffic_profile == "high_load":
            self.cpu_baseline_mean = min(1.0, self.cpu_baseline_mean * 5.0)
            self.ram_baseline_mean = min(1.0, self.ram_baseline_mean * 3.0)
            self.lambda_arrival = stats["arrival_rate"]["lambda_bursty"]
        elif self.traffic_profile == "bursty":
            self.spike_prob = stats["arrival_rate"]["spike_prob"] * 4.0
            self.failure_frequency = 0.05

LowTrafficConfig = EnvConfig(traffic_profile="steady")
HighTrafficConfig = EnvConfig(traffic_profile="high_load")
BurstyTrafficConfig = EnvConfig(traffic_profile="bursty")

# Helper map
CONFIGS = {
    "LowTraffic": LowTrafficConfig,
    "HighTraffic": HighTrafficConfig,
    "BurstyTraffic": BurstyTrafficConfig
}


def make_config(name: str, **overrides) -> EnvConfig:
    """A fresh EnvConfig for a named profile, so callers never mutate the shared CONFIGS.
    (dataclasses.replace would re-run __post_init__ and scale the High profile twice.)"""
    return EnvConfig(traffic_profile=CONFIGS[name].traffic_profile, **overrides)


# --- v2 environment (environment/cloud_env_v2.py) ---
# Fixed episode seeds shared by every method: VAL_SEEDS select checkpoints / tune baselines,
# evaluate_v2.TEST_SEEDS (10000-10099) are used only for the reported results.
VAL_SEEDS = range(9000, 9020)
MODEL_DIR = os.path.join("models", "v2")
RESULT_DIR = os.path.join("results", "v2")

# Load is measured in replica-capacity units: cpu = load / replicas.

@dataclass(frozen=True)
class CloudConfig:
    # Load process: daily cycle x mean-reverting (OU) noise x surge multiplier
    base_load: float = 2.5
    daily_amp: float = 0.3          # relative amplitude of the daily cycle
    daily_period: int = 288         # steps per day (5-minute steps)
    noise: float = 0.08             # OU noise scale (relative)
    ou_theta: float = 0.1           # OU mean-reversion rate

    # Fault scenarios: per-step probability that a fault starts
    p_leak: float = 0.0             # memory leak, fixed by restart
    p_err: float = 0.0              # error burst, fixed by clear cache (or restart)
    p_surge: float = 0.0            # traffic surge, handled by scaling
    leak_rate: float = 0.02         # memory fraction added per step while leaking
    surge_min: float = 1.5
    surge_max: float = 2.5
    surge_ramp: int = 3             # steps for a surge to reach its peak
    surge_len: tuple = (10, 30)

    # Cluster
    max_replicas: int = 12
    init_target_cpu: float = 0.6    # initial replicas sized for this utilisation
    provision_delay: int = 2        # steps before a scaled-up replica serves traffic

    # SLA
    sla_latency: float = 3.0        # latency in units of unloaded latency (~cpu < 0.67)
    sla_err: float = 0.05

    # Crash: OOM, or sustained heavy overload. A crash ends the episode.
    crash_cpu: float = 1.5
    crash_overload_steps: int = 4
    crash_penalty: float = 20.0

    # Reward = 1[SLA met] - c_rep*replicas/max - action cost - flapping penalty
    c_rep: float = 0.5
    c_restart: float = 0.3
    c_scale: float = 0.05
    c_cache: float = 0.02
    cold_cache_steps: int = 2       # after a cache clear, requests miss the cache for this many steps
    cold_cache_load: float = 1.15   # effective load multiplier while the cache is cold

    # Flapping: a scale reversal, or a second restart, inside flap_window steps
    flap_window: int = 10
    c_flap: float = 0.5

    # Ablation flags (metrics are recorded either way)
    cost_aware: bool = True
    anti_flapping: bool = True

    max_steps: int = 200


SCENARIOS = {
    "Low":        CloudConfig(base_load=1.5),
    "High":       CloudConfig(base_load=4.5),
    "Bursty":     CloudConfig(base_load=2.5, noise=0.15, p_surge=0.03),
    "MemLeak":    CloudConfig(base_load=2.5, p_leak=0.03),
    "ErrorBurst": CloudConfig(base_load=2.5, p_err=0.03),
    "Surge":      CloudConfig(base_load=2.5, p_surge=0.04),
    "Mixed":      CloudConfig(base_load=3.0, noise=0.12, p_leak=0.015, p_err=0.015, p_surge=0.02),
}
