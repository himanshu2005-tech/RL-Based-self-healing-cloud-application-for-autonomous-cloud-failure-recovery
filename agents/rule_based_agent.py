import numpy as np

class RuleBasedAgent:
    """
    A simple threshold-based policy representing traditional manual monitoring/scripting.
    State: [cpu_usage, ram_usage, response_time, error_rate, request_load]
    Actions: 0=restart, 1=scale_up, 2=scale_down, 3=clear_cache, 4=do_nothing
    """
    
    def __init__(self):
        pass
        
    def act(self, state, info=None):
        cpu_usage, ram_usage, response_time, error_rate, request_load = state
        
        if cpu_usage > 0.90 or error_rate > 0.30:
            return 0  # Restart
        elif cpu_usage > 0.75 or ram_usage > 0.85:
            return 1  # Scale up
        elif cpu_usage < 0.30 and ram_usage < 0.30:
            return 2  # Scale down
        elif response_time > 3.0:
            return 3  # Clear cache
        else:
            return 4  # Do nothing
