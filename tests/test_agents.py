import numpy as np

from agents.baselines import ThresholdAgent
from agents.q_learning_agent import QLearningAgent
from agents.rule_based_agent import RuleBasedAgent
from environment.cloud_env_v2 import RESTART, SCALE_UP, SCALE_DOWN, CLEAR_CACHE, HOLD


def obs(replicas=4, pending=0, cpu=0.5, mem=0.4, err=0.0, max_replicas=12):
    o = np.zeros(11, dtype=np.float32)
    o[0], o[1] = replicas / max_replicas, pending / max_replicas
    o[2], o[3], o[5] = cpu / 2, mem / 1.2, err
    return o


def test_threshold_agent_heals_and_scales():
    agent = ThresholdAgent(up=0.7, down=0.3, cooldown=0)
    assert agent.act(obs(mem=0.85)) == RESTART
    assert agent.act(obs(err=0.2, cpu=0.5)) == CLEAR_CACHE
    assert agent.act(obs(err=0.2, cpu=0.95)) == SCALE_UP      # errors from overload: scale, don't clear
    assert agent.act(obs(cpu=0.5)) == HOLD
    assert agent.act(obs(cpu=0.2)) == SCALE_DOWN


def test_threshold_agent_counts_pending_replicas():
    agent = ThresholdAgent(up=0.7)
    # 4 serving at 80% cpu, 1 pending -> projected 64%, no further scale-up
    assert agent.act(obs(replicas=4, pending=1, cpu=0.8)) == HOLD


def test_threshold_agent_respects_cooldown():
    agent = ThresholdAgent(up=0.7, down=0.3, cooldown=3)
    agent.reset()
    assert agent.act(obs(cpu=0.2)) == SCALE_DOWN
    assert agent.act(obs(cpu=0.2)) == HOLD


def test_v1_qlearning_uses_every_bin():
    agent = QLearningAgent(num_bins=5)
    idx = {agent._discretize_state([x, x, 10 * x, x, x])[0] for x in np.linspace(0, 1, 101)}
    assert idx == {0, 1, 2, 3, 4}


def test_v1_rule_defaults_match_original():
    agent = RuleBasedAgent()
    assert agent.act([0.95, 0.1, 1.0, 0.0, 0.5]) == 0
    assert agent.act([0.8, 0.1, 1.0, 0.0, 0.5]) == 1
    assert agent.act([0.2, 0.2, 1.0, 0.0, 0.5]) == 2
    assert agent.act([0.5, 0.5, 4.0, 0.0, 0.5]) == 3
    assert agent.act([0.5, 0.5, 1.0, 0.0, 0.5]) == 4
