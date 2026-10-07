"""
Significance tests and figures for the v2 results. Run after evaluate_v2.py.

    python analyze_v2.py

Writes results/v2/significance.csv, plots/v2/learning_curves.png and
plots/v2/gain_vs_best_rule.png.

Tests, per scenario, PPO against every other agent:
  * seed level (the unit of replication for a training algorithm): Welch t-test
    on per-seed mean rewards against another learned algorithm, or a one-sample
    t-test against a deterministic baseline's mean on the same episodes;
  * episode level: Wilcoxon signed-rank on the paired test episodes (each agent
    averaged over its seeds);
  * a 95% bootstrap CI of the mean reward difference, resampling seeds and
    episodes together.
p-values are Holm-corrected across all comparisons within each test.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from configs import RESULT_DIR

PLOT_DIR = os.path.join("plots", "v2")
REFERENCE = "PPO"
OTHERS = ["A2C", "DQN", "QLearning", "Tuned Threshold", "Tuned HPA+Heal", "HPA+Heal"]
LEARNED = ["PPO", "A2C", "DQN", "QLearning"]
BEST_RULE = "Tuned HPA+Heal"

# Chart style: validated categorical slots 1-3 (blue, orange, aqua) on the light surface
COLORS = {"PPO": "#2a78d6", "DQN": "#eb6834", "QLearning": "#1baf7a", "A2C": "#eda100"}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def bootstrap_diff(a, b, rng, n=5000):
    """a, b: arrays [seeds, episodes] of rewards on the same episodes. Resample seeds and episodes."""
    diffs = np.empty(n)
    n_ep = a.shape[1]
    for k in range(n):
        ep = rng.integers(n_ep, size=n_ep)
        sa = a[rng.integers(a.shape[0], size=a.shape[0])][:, ep]
        sb = b[rng.integers(b.shape[0], size=b.shape[0])][:, ep]
        diffs[k] = sa.mean() - sb.mean()
    return np.percentile(diffs, [2.5, 97.5])


def default_pairs(env_name, algos):
    return [(REFERENCE, other) for other in OTHERS if other in algos]


def trace_pairs(env_name, algos):
    pairs = [(REFERENCE, o) for o in ["A2C", "DQN", "QLearning", "RLlib PPO", "Tuned Threshold", "Tuned HPA+Heal"]]
    if env_name == "TraceFaults":
        pairs += [
            ("MA separate+comm", "MA separate"),        # does communication help?
            ("MA shared+comm", "MA shared"),
            ("MA shared+comm", "MA separate+comm"),     # shared vs separate policies
            ("MA shared", "MA separate"),
            ("MA separate+comm", "RLlib PPO"),          # multi-agent vs single-agent (same framework)
            ("MA shared+comm", "RLlib PPO"),
            ("MA separate+comm", "Tuned HPA+Heal"),
            ("MA shared+comm", "Tuned HPA+Heal"),
        ]
    return [(a, b) for a, b in pairs if a in algos and b in algos]


def significance(df, pair_fn=default_pairs):
    """Paired comparisons A vs B per scenario. A single-seed agent is treated as deterministic."""
    rng = np.random.default_rng(0)
    rows = []
    for env_name, g in df.groupby("Environment", sort=False):
        def matrix(algo):
            m = g[g.Algorithm == algo].pivot_table(index="train_seed", columns="episode_seed", values="reward")
            return m.sort_index(axis=1)
        for a, b in pair_fn(env_name, set(g.Algorithm)):
            ma, mb = matrix(a), matrix(b)
            sa, sb = ma.mean(axis=1).values, mb.mean(axis=1).values
            if len(sa) > 1 and len(sb) > 1:
                p_seed, seed_test = stats.ttest_ind(sa, sb, equal_var=False).pvalue, "Welch t"
            elif len(sb) == 1:
                p_seed, seed_test = stats.ttest_1samp(sa, sb.mean()).pvalue, "1-sample t"
            else:
                p_seed, seed_test = stats.ttest_1samp(sb, sa.mean()).pvalue, "1-sample t"
            p_ep = stats.wilcoxon(ma.mean(axis=0).values, mb.mean(axis=0).values).pvalue
            lo, hi = bootstrap_diff(ma.values, mb.values, rng)
            rows.append({
                "Environment": env_name, "Comparison": f"{a} vs {b}",
                "A Mean": ma.values.mean(), "B Mean": mb.values.mean(),
                "Diff": ma.values.mean() - mb.values.mean(), "CI Low": lo, "CI High": hi,
                "Seeds (A/B)": f"{len(sa)}/{len(sb)}",
                "Seed Test": seed_test, "p Seed": p_seed, "p Episode": p_ep,
            })
    out = pd.DataFrame(rows)
    out["p Seed (Holm)"] = holm(out["p Seed"])
    out["p Episode (Holm)"] = holm(out["p Episode"])
    out["Significant (both, 0.05)"] = (out["p Seed (Holm)"] < 0.05) & (out["p Episode (Holm)"] < 0.05)
    return out


def best_rules():
    """Per scenario, the tuned rule with the higher validation reward (chosen without test data)."""
    from tune_baselines import load_tuned
    return {env: "Tuned " + max(r, key=lambda k: r[k]["val_reward"]) for env, r in load_tuned().items()}


def best_rule_validation():
    from tune_baselines import load_tuned
    return {env: max(v["val_reward"] for v in r.values()) for env, r in load_tuned().items()}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


FILE_PREFIX = {"RLlib PPO": "RLlibPPO", "MA separate": "MA-separate-nocomm", "MA separate+comm": "MA-separate-comm",
               "MA shared": "MA-shared-nocomm", "MA shared+comm": "MA-shared-comm"}


def plot_learning_curves(envs, algos=LEARNED, colors=COLORS, fname="learning_curves.png",
                         title="Validation reward during training (20 fixed validation episodes)"):
    rule_val = best_rule_validation()
    ncols = 4 if len(envs) > 2 else len(envs) + 1
    nrows = 2 if len(envs) > 3 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.75 * ncols, 3.5 * nrows), facecolor=SURFACE, squeeze=False)
    for ax, env_name in zip(axes.flat, envs):
        style(ax)
        ends = []
        for algo in algos:
            prefix = FILE_PREFIX.get(algo, algo)
            files = sorted(glob.glob(os.path.join(RESULT_DIR, f"{prefix}_{env_name}_seed[0-9]_val.csv")))
            if not files:
                continue
            curves = [pd.read_csv(f) for f in files]
            n = min(len(c) for c in curves)
            x = curves[0].timestep.values[:n] / 1000
            y = np.stack([c.val_reward.values[:n] for c in curves])
            ax.fill_between(x, y.min(0), y.max(0), color=colors[algo], alpha=0.15, linewidth=0)
            ax.plot(x, y.mean(0), color=colors[algo], linewidth=2, label=algo)
            ends.append([y.mean(0)[-1], x[-1], algo])
        # Direct labels at line ends, pushed apart so they do not overlap
        lo, hi = ax.get_ylim()
        gap = 0.06 * (hi - lo)
        ends.sort(key=lambda e: -e[0])
        for i in range(1, len(ends)):
            ends[i][0] = min(ends[i][0], ends[i - 1][0] - gap)
        for y_end, x_end, algo in ends:
            ax.annotate(algo, (x_end, y_end), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=INK_2, va="center")
        ax.axhline(rule_val[env_name], color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), label="best tuned rule")
        ax.set_title(env_name, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("environment steps (thousands)", fontsize=8, color=MUTED)
        ax.set_ylabel("validation reward", fontsize=8, color=MUTED)
    for ax in list(axes.flat)[len(envs):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False, fontsize=10, labelcolor=INK_2,
                         title="mean of seeds, band = min to max", title_fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_gain(sig, n_episodes):
    fig, ax = plt.subplots(figsize=(9, 5.6), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    envs = list(dict.fromkeys(sig.Environment))
    algos = ["PPO", "A2C", "DQN", "QLearning"]
    # Gain over the best tuned rule = (PPO - rule) - (PPO - other)
    rules = best_rules()
    base = pd.concat([sig[(sig.Environment == e) & (sig.Comparison == f"PPO vs {rules[e]}")] for e in envs])
    base = base.set_index("Environment")
    for k, algo in enumerate(algos):
        ys = np.arange(len(envs)) + (k - 1.5) * 0.18
        if algo == "PPO":
            mid, lo, hi = base.Diff, base["CI Low"], base["CI High"]
        else:
            o = sig[sig.Comparison == f"PPO vs {algo}"].set_index("Environment")
            mid = base.Diff - o.Diff
            lo, hi = mid, mid  # CI shown for PPO only
        mid = mid.reindex(envs)
        ax.scatter(mid, ys, s=36, color=COLORS[algo], edgecolor=SURFACE, linewidth=1.5, zorder=3, label=algo)
        if algo == "PPO":
            ax.hlines(ys, lo.reindex(envs), hi.reindex(envs), color=COLORS[algo], linewidth=2, zorder=2)
    ax.axvline(0, color=AXIS, linewidth=1.2)
    ax.set_yticks(np.arange(len(envs)), [f"{e}\n(vs {rules[e][6:]})" for e in envs], color=INK_2, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"mean test reward minus the best tuned rule (same {n_episodes} episodes); PPO line = 95% bootstrap CI",
                  fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4)
    ax.set_title("Reward gain over the best tuned rule-based baseline", fontsize=12, color=INK, loc="left", pad=28)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "gain_vs_best_rule.png"), dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_ranking(df, env_name, fname):
    """Every agent's mean test reward minus the best tuned rule, with a 95% bootstrap CI."""
    rng = np.random.default_rng(0)
    g = df[df.Environment == env_name]
    rule = best_rules()[env_name]
    if rule not in set(g.Algorithm):
        rule = next(a for a in ["Tuned HPA+Heal", "Tuned Threshold", "HPA+Heal"] if a in set(g.Algorithm))

    def matrix(algo):
        return g[g.Algorithm == algo].pivot_table(index="train_seed", columns="episode_seed",
                                                  values="reward").sort_index(axis=1)
    ref = matrix(rule)
    rows = []
    for algo in dict.fromkeys(g.Algorithm):
        if algo in (rule, "Hold", "HPA"):
            continue
        m = matrix(algo)
        lo, hi = bootstrap_diff(m.values, ref.values, rng, n=2000)
        rows.append((algo, m.values.mean() - ref.values.mean(), lo, hi))
    rows.sort(key=lambda r: r[1], reverse=True)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.3), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ys = np.arange(len(rows))
    ax.hlines(ys, [r[2] for r in rows], [r[3] for r in rows], color=COLORS["PPO"], linewidth=2)
    ax.scatter([r[1] for r in rows], ys, s=36, color=COLORS["PPO"], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for y, r in zip(ys, rows):
        ax.annotate(f"{r[1]:+.1f}", (r[3], y), xytext=(5, 0), textcoords="offset points", fontsize=8,
                    color=INK_2, va="center")
    ax.axvline(0, color=AXIS, linewidth=1.2)
    ax.set_yticks(ys, [r[0] for r in rows], color=INK_2, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"mean test reward minus {rule} (95% bootstrap CI)", fontsize=8, color=MUTED)
    ax.set_title(f"{env_name}: every agent vs the best tuned rule", fontsize=12, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=130, facecolor=SURFACE)
    plt.close(fig)


# Separate figures, so each keeps <= 5 series in the validated categorical order
MA_COLORS = {"MA separate": "#2a78d6", "MA separate+comm": "#eb6834", "MA shared": "#1baf7a",
             "MA shared+comm": "#eda100", "RLlib PPO": "#e87ba4"}
TRACE_COLORS = {**COLORS, "RLlib PPO": "#e87ba4"}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="", help="analyse results/v2/evaluation_<tag>_episodes.csv")
    args = parser.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    os.makedirs(PLOT_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(RESULT_DIR, f"evaluation{suffix}_episodes.csv"))
    sig = significance(df, trace_pairs if args.tag == "trace" else default_pairs)
    sig.round(4).to_csv(os.path.join(RESULT_DIR, f"significance{suffix}.csv"), index=False)
    pd.set_option("display.width", 250)
    show = sig[["Environment", "Comparison", "Diff", "CI Low", "CI High", "Seeds (A/B)",
                "p Seed (Holm)", "p Episode (Holm)", "Significant (both, 0.05)"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    envs = list(dict.fromkeys(df.Environment))
    if args.tag == "trace":
        plot_learning_curves(envs, ["PPO", "A2C", "DQN", "QLearning", "RLlib PPO"], TRACE_COLORS,
                             "trace_learning_curves.png", "Trace scenarios: validation reward during training")
        if "TraceFaults" in envs:
            plot_learning_curves(["TraceFaults"], list(MA_COLORS), MA_COLORS, "multi_agent_learning_curves.png",
                                 "Multi-agent (scaler + healer) on TraceFaults: validation reward")
        for env_name in envs:
            plot_ranking(df, env_name, f"ranking_{env_name}.png")
    else:
        plot_learning_curves(envs)
        plot_gain(sig, df.episode_seed.nunique())
    print(f"figures written to {PLOT_DIR}")


if __name__ == "__main__":
    main()
