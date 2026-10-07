import numpy as np

# Original hand-set thresholds. In the v1 env steady-state response time is about 10 x cpu,
# so "response_time > 3.0" fires whenever cpu > 0.3 and the rule clears the cache ~86% of the time.
DEFAULT_THRESHOLDS = dict(restart_cpu=0.90, restart_err=0.30, scale_up_cpu=0.75, scale_up_ram=0.85,
                          scale_down=0.30, cache_rt=3.0)


class RuleBasedAgent:
    """
    A simple threshold-based policy representing traditional manual monitoring/scripting.
    State: [cpu_usage, ram_usage, response_time, error_rate, request_load]
    Actions: 0=restart, 1=scale_up, 2=scale_down, 3=clear_cache, 4=do_nothing
    """

    def __init__(self, **thresholds):
        self.t = {**DEFAULT_THRESHOLDS, **thresholds}

    def act(self, state, info=None):
        cpu_usage, ram_usage, response_time, error_rate, request_load = state
        t = self.t
        if cpu_usage > t["restart_cpu"] or error_rate > t["restart_err"]:
            return 0  # Restart
        elif cpu_usage > t["scale_up_cpu"] or ram_usage > t["scale_up_ram"]:
            return 1  # Scale up
        elif cpu_usage < t["scale_down"] and ram_usage < t["scale_down"]:
            return 2  # Scale down
        elif response_time > t["cache_rt"]:
            return 3  # Clear cache
        else:
            return 4  # Do nothing
