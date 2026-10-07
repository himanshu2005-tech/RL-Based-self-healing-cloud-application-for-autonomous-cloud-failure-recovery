# RL-Based Self-Healing Cloud Application

Reinforcement-learning agents that keep a simulated replicated cloud service healthy:
they scale it up and down with the load and repair faults (memory leaks, error bursts,
traffic surges) while meeting an SLA at the lowest replica cost and without flapping.

The project has two environments:

| | v1 (original) | v2 (current) |
|---|---|---|
| Code | `environment/cloud_env.py`, `train_*.py`, `evaluate.py` | `environment/cloud_env_v2.py`, `train_v2.py`, `evaluate_v2.py`, `analyze_v2.py` |
| Status | Frozen as the "before" reference | Used for all new results |

v1 was replaced after a review showed that in v1 a one-line rule ("restart if cpu > 0.6")
matched every trained agent (see [v1 results](#v1-results-original-environment)).

## Setup

Requires Git LFS (models, datasets and result CSVs are stored in LFS) and Python 3.11–3.13.

```powershell
git lfs install
git clone <repo-url> rl-selfheal
cd rl-selfheal
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest                      # 25 environment and agent tests
```

## Reproducing the results

```powershell
# Train PPO, A2C, DQN and Q-learning on all 7 scenarios x 5 seeds, plus the PPO ablation,
# tune the rule baselines, then evaluate everything on the shared test episodes
python run_v2_experiments.py --seeds 1 2 3 4 5 --jobs 6 --ablation-env Mixed
# Significance tests and figures
python analyze_v2.py
```

- **Single training run:** `python train_v2.py --algo PPO --env Mixed --seed 1`
- **Tune the rule baselines only:** `python tune_baselines.py`
- **Outputs:** `models/v2/`, `results/v2/` and `plots/v2/`
- **Runtime:** the full grid takes about 6 hours with 6 parallel jobs on a 16-thread
  laptop CPU. DQN is the slowest algorithm.

## The v2 environment

A replicated service with a time-varying load, one decision every 5 simulated minutes,
200 steps per episode.

- **Load** follows a daily cycle with mean-reverting (Ornstein–Uhlenbeck) noise, measured
  in replica-capacity units. `cpu = load / replicas`, and latency follows an M/M/1-style
  curve in cpu.
- **Actions** (Discrete 5):

  | Action | Effect |
  |---|---|
  | Scale up | +1 replica, serving after a 2-step provisioning delay |
  | Scale down | −1 replica, immediately |
  | Restart | Rolling restart: clears a memory leak and errors; one replica is out for a step |
  | Clear cache | Clears an error burst; the cache is cold (+15% load) for 2 steps |
  | Hold | Nothing |

- **Faults.** Each fault has one intended remedy:

  | Fault | What happens | Remedy |
  |---|---|---|
  | Memory leak | Memory grows until it hits 100% (an OOM crash) | Restart |
  | Error burst | The error rate stays high | Clear cache |
  | Traffic surge | Load multiplied by 1.5–2.5 for 10–30 steps | Scale up |

- **Reward:**

  `r = 1[SLA met] − 0.5·replicas/12 − action cost − 0.5·1[flap]`

  The SLA is met when latency ≤ 3× its unloaded value and error rate ≤ 5%. A crash (OOM,
  or cpu > 1.5 for 4 steps) ends the episode with −20. *Flapping* means reversing the
  scaling direction, or restarting a second time, within 10 steps. The observation
  includes the flapping state, so the penalty is Markov.
- **Observation (11 values):**
  - replicas and pending replicas
  - cpu, memory, latency relative to the SLA, and error rate
  - load and load trend
  - steps since the last scale, the last scale direction, and steps since the last restart

**Scenarios** (`configs.py`): Low, High and Bursty traffic with no faults; MemLeak,
ErrorBurst and Surge with a single fault type; and Mixed, with all three.

## Agents

| Agent | Kind | Tuning |
|---|---|---|
| Hold | Never acts | – |
| HPA | Kubernetes Horizontal Pod Autoscaler rule: `desired = ceil(replicas · cpu / 0.6)`, 5-step scale-down stabilisation | Default settings |
| HPA+Heal | HPA, plus restart when memory > 0.8 and clear cache when errors > 5% without overload | Default settings |
| Tuned HPA+Heal | HPA+Heal with target cpu, stabilisation window and heal thresholds tuned per scenario | Grid search, 840 configurations |
| Tuned Threshold | Scale up/down at cpu thresholds with a cooldown, plus the same heal rules | Grid search, 4,320 configurations |
| QLearning | Tabular Q-learning over 6 discretised features | – |
| DQN | Stable-Baselines3 DQN, 256×256 network, 300k steps | Settings chosen by a validation sweep |
| A2C | Stable-Baselines3 A2C (8 parallel envs, observation and reward normalisation), 400k steps | – |
| PPO | Stable-Baselines3 PPO (8 parallel envs, observation and reward normalisation), 400k steps | – |

**Same data and the same selection rule for every method:**
- **Validation (20 fixed episodes, seeds 9000–9019).** Each learned agent keeps its best
  checkpoint on these episodes, and each rule baseline is grid-searched on the same ones.
- **Test (100 fixed episodes, seeds 10000–10099).** Every agent is scored on these, and
  they are used for nothing else.
- **Training** uses different episodes again.

## Results (v2, 5 training seeds × 100 test episodes)

### Reward

Mean ± std over all test episodes, pooled across seeds. The rule baselines are
deterministic. The best mean in each row is in bold.

| Scenario | Hold | HPA | HPA+Heal | Tuned Threshold | Tuned HPA+Heal | QLearning | DQN | A2C | PPO |
|---|---|---|---|---|---|---|---|---|---|
| Low | 131.0 ± 48.8 | 166.8 ± 3.9 | 166.8 ± 3.9 | 169.0 ± 2.8 | **169.6 ± 2.8** | 162.4 ± 3.3 | 169.2 ± 3.4 | 168.2 ± 3.3 | 169.2 ± 3.1 |
| High | 74.2 ± 41.8 | 111.2 ± 8.1 | 111.2 ± 8.1 | 117.8 ± 7.0 | **118.1 ± 7.2** | 104.4 ± 8.9 | 115.6 ± 8.3 | 116.4 ± 8.3 | 118.1 ± 7.9 |
| Bursty | 53.8 ± 48.9 | 100.9 ± 13.9 | 100.9 ± 13.9 | 112.6 ± 13.6 | 112.7 ± 13.9 | 98.6 ± 26.1 | 111.8 ± 14.9 | 114.0 ± 14.5 | **115.5 ± 14.3** |
| MemLeak | 22.7 ± 24.4 | 27.7 ± 24.9 | 144.7 ± 5.4 | **150.6 ± 4.4** | 150.1 ± 4.1 | 139.5 ± 5.1 | 147.7 ± 10.8 | 147.9 ± 15.4 | 149.7 ± 9.3 |
| ErrorBurst | −11.2 ± 29.5 | −13.0 ± 32.5 | 139.9 ± 6.2 | **145.0 ± 5.0** | 144.7 ± 5.1 | 131.9 ± 7.0 | 143.4 ± 5.8 | 143.7 ± 5.0 | 144.3 ± 5.5 |
| Surge | 54.4 ± 46.2 | 112.6 ± 11.8 | 112.6 ± 11.8 | 119.3 ± 10.9 | 117.9 ± 10.7 | 105.4 ± 15.3 | 117.9 ± 11.5 | 118.9 ± 10.6 | **120.3 ± 10.5** |
| Mixed | −10.9 ± 24.0 | −12.9 ± 31.9 | 98.1 ± 16.7 | 105.4 ± 15.5 | 106.3 ± 15.7 | 85.1 ± 30.5 | 100.4 ± 20.3 | 104.9 ± 15.9 | **107.3 ± 14.9** |

### Operational metrics

**SLA violation % / crashes per episode / flaps per episode:**

| Scenario | HPA+Heal | Tuned Threshold | Tuned HPA+Heal | QLearning | DQN | A2C | PPO |
|---|---|---|---|---|---|---|---|
| Low | 1.8 / 0.00 / 5.6 | 1.1 / 0.00 / 1.5 | 0.8 / 0.00 / 1.2 | 0.2 / 0.00 / 2.3 | 1.2 / 0.00 / 0.1 | 1.1 / 0.00 / 0.5 | 1.2 / 0.00 / 0.5 |
| High | 5.3 / 0.00 / 13.9 | 2.3 / 0.00 / 1.8 | 2.8 / 0.00 / 2.8 | 3.1 / 0.00 / 4.2 | 4.4 / 0.00 / 0.7 | 3.8 / 0.00 / 3.0 | 3.1 / 0.00 / 2.5 |
| Bursty | 17.4 / 0.00 / 15.0 | 6.7 / 0.00 / 2.3 | 8.0 / 0.00 / 2.6 | 15.7 / 0.04 / 13.9 | 8.7 / 0.00 / 3.7 | 8.0 / 0.00 / 4.8 | 8.0 / 0.00 / 3.4 |
| MemLeak | 3.3 / 0.00 / 11.1 | 1.3 / 0.00 / 1.0 | 1.2 / 0.00 / 1.4 | 0.5 / 0.00 / 4.5 | 2.5 / 0.02 / 0.4 | 1.8 / 0.03 / 0.7 | 2.1 / 0.01 / 1.7 |
| ErrorBurst | 6.1 / 0.00 / 10.3 | 4.2 / 0.00 / 0.9 | 4.2 / 0.00 / 1.4 | 4.6 / 0.00 / 2.3 | 5.6 / 0.00 / 0.4 | 4.1 / 0.00 / 2.3 | 5.0 / 0.00 / 0.8 |
| Surge | 12.4 / 0.00 / 11.2 | 7.1 / 0.00 / 2.0 | 6.1 / 0.00 / 2.1 | 7.4 / 0.00 / 13.1 | 7.9 / 0.00 / 1.9 | 5.9 / 0.00 / 4.9 | 5.7 / 0.00 / 3.4 |
| Mixed | 16.7 / 0.00 / 14.2 | 11.3 / 0.00 / 5.7 | 10.0 / 0.00 / 3.2 | 16.4 / 0.13 / 11.4 | 16.1 / 0.04 / 2.8 | 9.9 / 0.00 / 7.9 | 9.8 / 0.00 / 3.6 |

**Mean replicas:**

| Scenario | HPA+Heal | Tuned Threshold | Tuned HPA+Heal | QLearning | DQN | A2C | PPO |
|---|---|---|---|---|---|---|---|
| Low | 3.13 | 3.30 | 3.34 | 4.27 | 3.37 | 3.40 | 3.32 |
| High | 8.33 | 9.12 | 8.88 | 10.41 | 8.93 | 8.73 | 8.82 |
| Bursty | 6.46 | 8.60 | 8.20 | 7.14 | 8.02 | 7.82 | 7.79 |
| MemLeak | 4.91 | 5.39 | 5.42 | 6.62 | 5.33 | 5.21 | 5.11 |
| ErrorBurst | 4.96 | 5.47 | 5.47 | 6.76 | 5.34 | 5.47 | 5.32 |
| Surge | 6.56 | 7.73 | 8.11 | 8.53 | 7.65 | 7.76 | 7.78 |
| Mixed | 6.99 | 7.99 | 8.36 | 7.99 | 7.29 | 8.08 | 8.19 |

Full table, including action distributions: `results/v2/evaluation_summary.csv`.
Per-episode rows: `results/v2/evaluation_episodes.csv`.
Tuned rule parameters: `results/v2/tuned_baselines.json`.

### Significance (PPO vs each agent)

Each comparison uses two tests:
- **Seed level:** Welch t-test on per-seed means against another learned agent, or a
  one-sample t-test against a deterministic baseline.
- **Episode level:** paired Wilcoxon test on the same test episodes.

Both are Holm-corrected across all 42 comparisons. "Significant" means p < 0.05 in both
tests. The CI is a 95% bootstrap over seeds and episodes.

| Scenario | Comparison | Reward diff [95% CI] | p seed | p episode | Significant |
|---|---|---|---|---|---|
| Low | PPO vs A2C | +1.0 [0.3, 1.6] | 0.61 | <0.0001 | no |
| Low | PPO vs DQN | −0.0 [−0.4, 0.4] | 1 | 1 | no |
| Low | PPO vs QLearning | +6.8 [4.3, 9.5] | 0.21 | <0.0001 | no |
| Low | PPO vs Tuned Threshold | +0.2 [−0.4, 0.7] | 1 | 1 | no |
| Low | PPO vs Tuned HPA+Heal | −0.4 [−0.8, 0.1] | 0.78 | 0.25 | no |
| Low | PPO vs HPA+Heal | +2.3 [1.7, 3.0] | 0.0033 | <0.0001 | **yes** |
| High | PPO vs A2C | +1.7 [0.5, 2.9] | 0.61 | <0.0001 | no |
| High | PPO vs DQN | +2.5 [1.4, 3.7] | 0.072 | <0.0001 | no |
| High | PPO vs QLearning | +13.7 [12.0, 15.3] | 0.0001 | <0.0001 | **yes** |
| High | PPO vs Tuned Threshold | +0.3 [−0.5, 1.1] | 1 | 1 | no |
| High | PPO vs Tuned HPA+Heal | +0.0 [−0.8, 0.8] | 1 | 1 | no |
| High | PPO vs HPA+Heal | +6.9 [5.9, 7.8] | 0.0014 | <0.0001 | **yes** |
| Bursty | PPO vs A2C | +1.6 [0.7, 2.5] | 0.24 | <0.0001 | no |
| Bursty | PPO vs DQN | +3.7 [2.4, 5.0] | 0.07 | <0.0001 | no |
| Bursty | PPO vs QLearning | +16.9 [13.2, 21.0] | 0.013 | <0.0001 | **yes** |
| Bursty | PPO vs Tuned Threshold | +3.0 [2.0, 3.9] | 0.0093 | <0.0001 | **yes** |
| Bursty | PPO vs Tuned HPA+Heal | +2.8 [1.9, 3.8] | 0.011 | <0.0001 | **yes** |
| Bursty | PPO vs HPA+Heal | +14.6 [13.4, 15.9] | <0.0001 | <0.0001 | **yes** |
| MemLeak | PPO vs A2C | +1.9 [−0.9, 5.2] | 1 | 1 | no |
| MemLeak | PPO vs DQN | +2.0 [−0.6, 4.6] | 1 | <0.0001 | no |
| MemLeak | PPO vs QLearning | +10.3 [6.8, 13.9] | 0.072 | <0.0001 | no |
| MemLeak | PPO vs Tuned Threshold | −0.9 [−3.1, 0.5] | 1 | 1 | no |
| MemLeak | PPO vs Tuned HPA+Heal | −0.4 [−2.4, 1.0] | 1 | 1 | no |
| MemLeak | PPO vs HPA+Heal | +5.1 [3.0, 6.6] | 0.07 | <0.0001 | no |
| ErrorBurst | PPO vs A2C | +0.6 [−0.4, 1.4] | 1 | 0.006 | no |
| ErrorBurst | PPO vs DQN | +0.9 [0.2, 1.6] | 0.61 | <0.0001 | no |
| ErrorBurst | PPO vs QLearning | +12.4 [9.3, 16.4] | 0.088 | <0.0001 | no |
| ErrorBurst | PPO vs Tuned Threshold | −0.8 [−1.5, −0.1] | 0.78 | 0.0031 | no |
| ErrorBurst | PPO vs Tuned HPA+Heal | −0.4 [−1.1, 0.2] | 1 | 0.8 | no |
| ErrorBurst | PPO vs HPA+Heal | +4.3 [3.5, 5.2] | 0.004 | <0.0001 | **yes** |
| Surge | PPO vs A2C | +1.4 [0.4, 2.6] | 0.61 | <0.0001 | no |
| Surge | PPO vs DQN | +2.4 [1.4, 3.6] | 0.04 | <0.0001 | **yes** |
| Surge | PPO vs QLearning | +14.9 [11.8, 18.3] | 0.021 | <0.0001 | **yes** |
| Surge | PPO vs Tuned Threshold | +1.0 [0.0, 2.2] | 0.78 | 0.36 | no |
| Surge | PPO vs Tuned HPA+Heal | +2.4 [1.5, 3.5] | 0.072 | <0.0001 | no |
| Surge | PPO vs HPA+Heal | +7.8 [6.6, 8.9] | 0.0013 | <0.0001 | **yes** |
| Mixed | PPO vs A2C | +2.4 [−0.6, 5.3] | 1 | 0.0002 | no |
| Mixed | PPO vs DQN | +6.9 [3.9, 10.3] | 0.07 | <0.0001 | no |
| Mixed | PPO vs QLearning | +22.1 [15.5, 29.2] | 0.07 | <0.0001 | no |
| Mixed | PPO vs Tuned Threshold | +1.9 [−0.2, 3.9] | 0.25 | 0.34 | no |
| Mixed | PPO vs Tuned HPA+Heal | +0.9 [−1.1, 3.1] | 1 | 1 | no |
| Mixed | PPO vs HPA+Heal | +9.2 [6.9, 11.4] | 0.0012 | <0.0001 | **yes** |

Full table: `results/v2/significance.csv`.

### Ablation: reward terms (PPO on Mixed, 5 seeds)

Each variant is trained without some reward terms, then **all variants are scored with
the full reward** on the same test episodes, so the rewards are comparable.

| Trained with | Reward | SLA violation % | Replicas | Flaps / ep | Crashes / ep |
|---|---|---|---|---|---|
| Full reward | **107.3 ± 14.9** | 9.8 | 8.19 | 3.65 | 0.00 |
| No action cost | 106.0 ± 15.6 | 9.4 | 8.20 | 5.90 | 0.00 |
| No flapping penalty | 90.8 ± 16.8 | 9.6 | 7.71 | 39.90 | 0.00 |
| Neither | 70.3 ± 18.2 | 10.2 | 7.50 | 72.07 | 0.00 |

### Figures

![Validation reward during training](plots/v2/learning_curves.png)

![Reward gain over the best tuned rule](plots/v2/gain_vs_best_rule.png)

### Findings

1. **PPO matches a per-scenario tuned rule and beats it only under bursty traffic.**
   - Against the best tuned rule, PPO's mean is within ±2 reward in 6 of 7 scenarios,
     with no significant difference.
   - PPO is significantly better in **Bursty**: +3.0 over Tuned Threshold and +2.8 over
     Tuned HPA+Heal. There, short random load spikes reward anticipating demand, which
     fixed thresholds can't do.
   - Against untuned defaults (HPA+Heal) PPO gains +2.3 to +14.6, significant in 6 of 7
     scenarios. Most of that gain is also available from tuning the rule's thresholds.
2. **What RL buys is reuse, not peak reward.** The same PPO configuration reaches
   rule-level performance in every scenario with no hand-designed logic. Each tuned rule
   needed a scenario-specific grid search over its parameters (840–4,320 configurations
   each), and the rule logic itself (which fault maps to which action) was designed by
   hand.
3. **The three deep RL methods are close.**
   - PPO vs A2C: no significant seed-level difference in any scenario.
   - PPO vs DQN: no significant difference in 6 of 7 (PPO wins in Surge).
   - PPO has the highest or tied-highest mean of the learned agents in all 7 scenarios,
     and a consistently small seed-to-seed spread (std of seed means 0.3–1.5).
   - A2C is close behind, with more flapping (up to 7.9 per episode on Mixed).
   - DQN needed a settings sweep to stop collapsing. On Mixed it went from 31.9 (crashes
     on 79% of episodes) to 100.4 (4%). It is still the weakest of the three deep RL
     methods on Mixed.
4. **Tabular Q-learning is clearly worst**, 7–22 reward below PPO. Its discretised state
   cannot represent load precisely, so it over-provisions.
5. **Each fault needs its own remedy.** HPA without healing crashes on 99% of MemLeak
   episodes, and violates the SLA on 83% of ErrorBurst steps.
6. **The flapping penalty is the reward term that matters.** Without it, flaps rise
   about 11× and reward drops 16.5 points. The action-cost term has little effect on
   its own.

**Implication for the next stages.** The synthetic scenarios are simple enough that a
well-tuned threshold rule is near-optimal. RL has more room to win with:
- non-stationary, trace-driven load (Borg trace)
- one policy across many scenarios, compared against rules tuned per scenario
- the multi-agent split

These are the next items on the roadmap.

### How the DQN settings were chosen

A sweep on MemLeak and Mixed (2 seeds, selected on validation reward only) compared four
settings:

| Setting | MemLeak best val | Mixed best val |
|---|---|---|
| Stage-B (lr 5e-4, target update 500, 64×64 net) | 148.0 / 144.9 | 31.0 / 26.9 |
| **Stable (lr 1e-4, target update 1000, 256×256 net, buffer 200k, batch 128, train_freq 4)** | 148.6 / 148.4 | **104.9 / 105.0** |
| Stable + longer exploration | 148.8 / 150.6 | 48.8 / 29.1 |
| Stable + γ = 0.995 | 147.6 / 149.5 | 65.6 / 28.7 |

### Baseline tuning range

The grids are listed in `tune_baselines.py`. Some tuned values sit on the edge of their
grid:
- **Heal thresholds** in scenarios without that fault. They have no effect there
  (changing them alters validation reward by 0.00), so the first grid value wins the tie.
- **Cooldown in Low and High.** One grid step inward changes validation reward by at most
  0.2.

So the tuned baselines are at their optimum for these rule families.

## v1 results (original environment)

Kept for reference (`python evaluate.py`; 3 seeds × 10 episodes; std across seed means,
or across episodes for the single-seed rule rows).

| Env | Algorithm | Original | After fixes | Notes |
|---|---|---|---|---|
| LowTraffic | RuleBased | 86.71 ± 0.00 | 86.71 ± 7.90 | Std is now over episodes (C2) |
| LowTraffic | RuleBased (tuned) | – | **121.51 ± 10.79** | Thresholds tuned on validation episodes (D1) |
| LowTraffic | QLearning | 95.74 ± 6.05 | 94.86 ± 10.15 | Bin edges fixed (D2) |
| LowTraffic | DQN | 77.23 ± 9.09 | 118.32 ± 1.44 | Hyperparameters fixed (A1) |
| LowTraffic | PPO | 119.95 ± 3.19 | 119.29 ± 3.79 | Retrained (torch version) |
| HighTraffic | RuleBased | 86.26 ± 0.00 | 86.26 ± 8.10 | |
| HighTraffic | RuleBased (tuned) | – | **121.30 ± 11.45** | |
| HighTraffic | QLearning | 100.14 ± 0.49 | 90.62 ± 4.07 | |
| HighTraffic | DQN | 76.62 ± 11.78 | 116.45 ± 2.69 | |
| HighTraffic | PPO | 117.17 ± 3.95 | 114.94 ± 3.77 | |
| BurstyTraffic | RuleBased | 73.92 ± 0.00 | 73.92 ± 1.80 | |
| BurstyTraffic | RuleBased (tuned) | – | **110.42 ± 2.05** | |
| BurstyTraffic | QLearning | 86.65 ± 4.53 | 93.60 ± 2.19 | |
| BurstyTraffic | DQN | 46.00 ± 5.32 | 108.02 ± 1.62 | |
| BurstyTraffic | PPO | 110.42 ± 0.21 | 110.18 ± 0.30 | |

**Why v1 was replaced:**
- The tuned v1 rule settles on "restart if cpu > 0.6, never clear the cache". It is the
  best or tied-best policy in every v1 profile.
- LowTraffic and HighTraffic are effectively the same environment (do-nothing scores
  70.6 vs 69.7).
- Restart is the only useful action.
- The v1 ablation table (`results/ablation_summary.csv`) is invalid. Each variant is
  scored with its own reward function, and the no-flapping variants report 0 flaps by
  construction.

**Other v1 fixes:**
- Training CSVs now record crashes and flapping from `info`. Before, they were always 0,
  because the environment had already been reset when they were read.
- Configs are built with `make_config()` instead of mutating the shared `CONFIGS`.

## Repository layout

```
configs.py                  v1 EnvConfig + make_config(); v2 CloudConfig, SCENARIOS, seeds and paths
environment/cloud_env.py    v1 environment (frozen)
environment/cloud_env_v2.py v2 environment
agents/baselines.py         Hold, HPA, HPA+Heal, ThresholdAgent (act on the observation only)
agents/q_learning_agent.py  tabular Q-learning (v1 defaults, custom bins for v2)
agents/rule_based_agent.py  v1 threshold rule (tunable thresholds)
train_v2.py                 train QLearning / DQN / A2C / PPO on a v2 scenario
tune_baselines.py           grid-search the v2 rule baselines on the validation episodes
evaluate_v2.py              shared-episode evaluation + reward ablation
analyze_v2.py               significance tests and figures
run_v2_experiments.py       parallel training grid, baseline tuning and evaluation
tune_rule_v1.py             tune the v1 rule thresholds
tests/                      pytest tests for the v2 environment and the agents
train_*.py, evaluate.py     v1 pipeline
data_analysis/              Borg trace calibration (v1)
```

## Roadmap

- [x] Fix DQN hyperparameters (v1)
- [x] v2 environment: replicas in state, capacity-based cpu, replica cost, SLA reward,
      time-windowed flapping, fault scenarios
- [x] Evaluation: shared test episodes, 5 seeds, valid ablation, significance tests,
      learning curves
- [x] Tuned threshold and HPA baselines, stable DQN, A2C; v1 rule thresholds, Q-learning
      bins and training metrics fixed
- [ ] Drive the load from the Google Borg trace, with a fixed train/test split
- [ ] Port to RLlib (needs gymnasium 1.x and stable-baselines3 ≥ 2.4)
- [ ] Multi-agent RLlib env: a scaler agent and a healer agent, with separate and shared
      policies
