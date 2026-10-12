#!/usr/bin/env python3
"""
L-DDT (Local-RTG Decision Transformer) result visualization.

Usage:
    python plot_results.py --data_dir ./csv --out_dir ./figs

Expected CSVs in --data_dir:
    evaluation_DDT_summary.csv, evaluation_DDT_paired_vs_DT.csv, evaluation_DDT_raw.csv,
    eval_main_summary.csv, B_k_metrics.csv, B_lag_effect.csv, B_summary.csv,
    C_rtg_signal.csv, D_rtg_vs_near_reward.csv

Missing files are skipped with a warning. Only pandas / numpy / matplotlib are required.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLLISIONS = [0.0, 0.3, 0.6, 0.9]
KS = [1, 5, 10, 20, 50]
DDT_MODELS = [f"DDT_k{k}" for k in KS]
CMAP = plt.get_cmap("viridis")
K_COLOR = {k: CMAP(i / (len(KS) - 1) * 0.85) for i, k in enumerate(KS)}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False,
})


def load(data_dir, name):
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        print(f"[skip] {name} not found")
        return None
    return pd.read_csv(path)


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


# ---------------------------------------------------------------- Fig 1
def fig_paired_diff(paired, out_dir):
    """Mean paired difference vs DT (95% CI) as a function of k, per collision."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True)
    for ax, c in zip(axes, COLLISIONS):
        d = paired[paired.collision == c].set_index("model")
        xs = list(range(len(KS)))
        ys = np.array([d.loc[m, "mean_diff"] for m in DDT_MODELS])
        ci = np.array([1.96 * d.loc[m, "sem"] for m in DDT_MODELS])
        sig = np.abs(ys) > ci
        ax.axhline(0, color="k", lw=1)
        ax.errorbar(xs, ys, yerr=ci, fmt="-", color="gray", alpha=0.6, capsize=0)
        for x, y, e, s in zip(xs, ys, ci, sig):
            ax.errorbar(x, y, yerr=e, fmt="o", capsize=4,
                        color="tab:red" if s else "tab:blue",
                        mfc="tab:red" if s else "white")
        if "greedy" in d.index:
            ax.axhline(d.loc["greedy", "mean_diff"], color="tab:orange", ls="--",
                       label="greedy")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"k={k}" for k in KS])
        ax.set_title(f"collision = {c}")
        if c == COLLISIONS[0]:
            ax.set_ylabel("Return difference vs DT (paired)")
            ax.legend(loc="best")
    fig.suptitle("DDT − DT per k (red filled: 95% CI excludes 0; dashed: greedy − DT)",
                 y=1.03)
    save(fig, out_dir, "fig1_paired_diff_vs_DT.png")


# ---------------------------------------------------------------- Fig 2
def fig_normalized_score(summary, out_dir):
    """Score normalized between random (0) and greedy (1)."""
    piv = summary.pivot(index="model", columns="collision", values="mean")
    models = ["DT"] + DDT_MODELS
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.13
    xs = np.arange(len(COLLISIONS))
    for i, m in enumerate(models):
        vals = [(piv.loc[m, c] - piv.loc["random", c]) /
                (piv.loc["greedy", c] - piv.loc["random", c]) for c in COLLISIONS]
        color = "gray" if m == "DT" else K_COLOR[int(m.split("k")[-1])]
        ax.bar(xs + (i - len(models) / 2 + 0.5) * width, vals, width,
               label=m, color=color)
    ax.axhline(1, color="tab:orange", ls="--", label="greedy")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"c={c}" for c in COLLISIONS])
    ax.set_ylabel("Normalized score  (random=0, greedy=1)")
    ax.set_title("Where each model sits between random and greedy")
    ax.legend(ncol=4, fontsize=9)
    save(fig, out_dir, "fig2_normalized_score.png")


# ---------------------------------------------------------------- Fig 3
def fig_win_rate(raw, out_dir):
    """Per-seed paired difference distribution and win rate (DDT_k - DT)."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    for ax, c in zip(axes, COLLISIONS):
        d = raw[raw.collision == c].pivot(index="seed", columns="model", values="ret")
        data, labels, wins = [], [], []
        for m in DDT_MODELS:
            diff = (d[m] - d["DT"]).values
            data.append(diff)
            labels.append(m.replace("DDT_", ""))
            wins.append((diff > 0).mean())
        bp = ax.boxplot(data, showfliers=False, patch_artist=True)
        ax.set_xticklabels(labels)
        for patch, k in zip(bp["boxes"], KS):
            patch.set_facecolor(K_COLOR[k])
            patch.set_alpha(0.6)
        ax.axhline(0, color="k", lw=1)
        top = ax.get_ylim()[1]
        for i, w in enumerate(wins, 1):
            ax.text(i, top, f"{w:.0%}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"collision = {c}", pad=18)
        if c == COLLISIONS[0]:
            ax.set_ylabel("Per-seed return diff vs DT")
    fig.suptitle("Per-seed paired difference (numbers on top: win rate over 50 seeds)",
                 y=1.04)
    save(fig, out_dir, "fig3_per_seed_diff_winrate.png")


# ---------------------------------------------------------------- Fig 4
def fig_rtg_sensitivity(main_summary, out_dir):
    """Target-RTG multiplier (x0.5 / x1.0 / x1.5) sensitivity: controllability."""
    s = main_summary.copy()
    piv = s.pivot(index="model", columns="collision", values="mean")
    models = ["DT"] + DDT_MODELS
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, c in zip(axes, COLLISIONS):
        for m in models:
            names = [f"{m}@x0.5", m, f"{m}@x1.5"]
            if not all(n in piv.index for n in names):
                continue
            y = np.array([piv.loc[n, c] for n in names])
            y = y - y[1]  # relative to x1.0
            color = "gray" if m == "DT" else K_COLOR[int(m.split("k")[-1])]
            ax.plot([0.5, 1.0, 1.5], y, "o-", color=color, label=m,
                    lw=2.5 if m in ("DT", "DDT_k1") else 1.5)
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks([0.5, 1.0, 1.5])
        ax.set_xlabel("Target-RTG multiplier")
        ax.set_title(f"collision = {c}")
        if c == COLLISIONS[0]:
            ax.set_ylabel("Return change vs x1.0")
            ax.legend(fontsize=8)
    fig.suptitle("Controllability: does changing the target RTG change performance?",
                 y=1.03)
    save(fig, out_dir, "fig4_rtg_controllability.png")

    # compact bar: range (x1.5 - x0.5)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.13
    xs = np.arange(len(COLLISIONS))
    for i, m in enumerate(models):
        names = [f"{m}@x0.5", f"{m}@x1.5"]
        if not all(n in piv.index for n in names):
            continue
        vals = [piv.loc[names[1], c] - piv.loc[names[0], c] for c in COLLISIONS]
        color = "gray" if m == "DT" else K_COLOR[int(m.split("k")[-1])]
        ax.bar(xs + (i - len(models) / 2 + 0.5) * width, vals, width,
               label=m, color=color)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"c={c}" for c in COLLISIONS])
    ax.set_ylabel("Return(x1.5) − Return(x0.5)")
    ax.set_title("RTG sensitivity (larger = target RTG actually steers the policy)")
    ax.legend(ncol=3, fontsize=9)
    save(fig, out_dir, "fig4b_rtg_sensitivity_range.png")


# ---------------------------------------------------------------- Fig 5
def fig_action_effect(lag, kmet, out_dir):
    """Experiment B: how long does an action's effect persist, and SNR vs k."""
    if lag is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for c in COLLISIONS:
            d = lag[lag.collision == c]
            axes[0].plot(d.lag, d.eff, label=f"c={c}", lw=2)
            axes[1].semilogy(d.lag, d.eff.clip(lower=1e-4), label=f"c={c}", lw=2)
        axes[0].set_title("Action effect vs lag")
        axes[1].set_title("Action effect vs lag (log scale)")
        for ax in axes:
            ax.set_xlabel("Lag (steps after the action)")
            ax.set_ylabel("Effect on reward")
            ax.legend()
        save(fig, out_dir, "fig5a_action_effect_lag.png")

    if kmet is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for c in COLLISIONS:
            d = kmet[kmet.collision == c]
            axes[0].plot(d.k, d.cum_eff, "o-", label=f"c={c}")
            axes[1].loglog(d.k, d.snr, "o-", label=f"c={c}")
            axes[2].plot(d.k, d.rank_acc, "o-", label=f"c={c}")
        axes[0].set_title("Cumulative action effect")
        axes[1].set_title("SNR of local RTG (effect / sd)")
        axes[2].set_title("Action ranking accuracy")
        axes[2].axhline(0.5, color="k", ls=":", lw=1)
        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlabel("Local RTG horizon k")
            ax.legend(fontsize=9)
        save(fig, out_dir, "fig5b_B_k_metrics.png")


# ---------------------------------------------------------------- Fig 6
def fig_rtg_signal(sig, out_dir):
    """Experiment C: does RTG carry information about the action?"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    ks = [1, 2, 3, 5, 10, 20, 50]
    for ax, c in zip(axes, COLLISIONS):
        d = sig[sig.collision == c].set_index("variant")
        loc = [d.loc[f"local_k{k}", "dR2"] for k in ks]
        lag = [d.loc[f"lagged_k{k}", "dR2"] for k in ks]
        rnd = [d.loc[f"random_k{k}", "dR2"] for k in ks]
        ax.plot(ks, loc, "o-", label="local", lw=2)
        ax.plot(ks, lag, "s--", label="lagged", alpha=0.7)
        ax.plot(ks, rnd, "^:", label="random", alpha=0.7)
        ax.axhline(d.loc["global", "dR2"], color="tab:red", ls="--",
                   label="global")
        ax.axhline(0, color="k", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("k")
        ax.set_title(f"collision = {c}")
        if c == COLLISIONS[0]:
            ax.set_ylabel("ΔR²  (adding action to state)")
            ax.legend(fontsize=9)
    fig.suptitle("Does RTG carry action information? (ΔR² = R²(s,a) − R²(s))", y=1.03)
    save(fig, out_dir, "fig6a_C_dR2.png")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, c in zip(axes, COLLISIONS):
        d = sig[sig.collision == c].set_index("variant")
        names = ["global"] + [f"local_k{k}" for k in ks]
        r_s = [d.loc[n, "R2_s"] for n in names]
        r_sa = [d.loc[n, "R2_sa"] for n in names]
        x = np.arange(len(names))
        ax.bar(x - 0.2, r_s, 0.4, label="R²(s)")
        ax.bar(x + 0.2, r_sa, 0.4, label="R²(s,a)")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("local_", "") for n in names],
                           rotation=45, fontsize=9)
        ax.set_title(f"collision = {c}")
        if c == COLLISIONS[0]:
            ax.set_ylabel("Variance of RTG explained")
            ax.legend(fontsize=9)
    fig.suptitle("RTG variance explained by state (s) vs state+action (s,a)", y=1.03)
    save(fig, out_dir, "fig6b_C_R2.png")


# ---------------------------------------------------------------- Fig 7
def fig_rtg_near_reward(d, out_dir):
    """Experiment D: correlation of RTG with near-future reward."""
    g = d[d.rtg == "global"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in COLLISIONS:
        x = g[g.collision == c].sort_values("near_k")
        ax.plot(x.near_k, x["corr"], "o-", label=f"c={c}", lw=2)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Near-future reward window (steps)")
    ax.set_ylabel("Corr(Global RTG, near reward)")
    ax.set_title("Global RTG explains little of the near-term reward")
    ax.legend()
    save(fig, out_dir, "fig7_D_global_rtg_vs_near_reward.png")


# ---------------------------------------------------------------- Fig 8
def fig_mean_return(summary, out_dir):
    """Raw mean return with SEM per model (context: differences are small)."""
    piv = summary.pivot(index="model", columns="collision", values="mean")
    sem = summary.pivot(index="model", columns="collision", values="sem")
    models = ["random", "greedy", "DT"] + DDT_MODELS
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for m in models:
        color = {"random": "lightgray", "greedy": "tab:orange", "DT": "gray"}.get(
            m, K_COLOR.get(int(m.split("k")[-1]), "C0") if m.startswith("DDT") else "C0")
        ax.errorbar(COLLISIONS, [piv.loc[m, c] for c in COLLISIONS],
                    yerr=[sem.loc[m, c] for c in COLLISIONS],
                    marker="o", label=m, color=color, capsize=3)
    ax.set_xlabel("Collision probability")
    ax.set_ylabel("Mean return (±SEM, 50 seeds)")
    ax.set_title("Absolute return: throughput is dominated by collision rate")
    ax.legend(ncol=4, fontsize=9)
    save(fig, out_dir, "fig8_mean_return.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--out_dir", default="./figs")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    summary = load(a.data_dir, "evaluation_DDT_summary.csv")
    paired = load(a.data_dir, "evaluation_DDT_paired_vs_DT.csv")
    raw = load(a.data_dir, "evaluation_DDT_raw.csv")
    main_sum = load(a.data_dir, "eval_main_summary.csv")
    kmet = load(a.data_dir, "B_k_metrics.csv")
    lag = load(a.data_dir, "B_lag_effect.csv")
    sig = load(a.data_dir, "C_rtg_signal.csv")
    near = load(a.data_dir, "D_rtg_vs_near_reward.csv")

    if paired is not None:
        fig_paired_diff(paired, a.out_dir)
    if summary is not None:
        fig_normalized_score(summary, a.out_dir)
        fig_mean_return(summary, a.out_dir)
    if raw is not None:
        fig_win_rate(raw, a.out_dir)
    if main_sum is not None:
        fig_rtg_sensitivity(main_sum, a.out_dir)
    if lag is not None or kmet is not None:
        fig_action_effect(lag, kmet, a.out_dir)
    if sig is not None:
        fig_rtg_signal(sig, a.out_dir)
    if near is not None:
        fig_rtg_near_reward(near, a.out_dir)


if __name__ == "__main__":
    main()
