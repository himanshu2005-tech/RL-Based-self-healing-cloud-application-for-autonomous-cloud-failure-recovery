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
