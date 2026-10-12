# p1_analyze.py
"""
P1 analysis: (A) collision grid, (B) target-RTG multiplier sweep.  No training, no env needed.

    python p1_analyze.py grid \
        --raw p1_grid_raw.csv --baselines p1_grid_baselines_raw.csv --out-prefix p1_grid

    python p1_analyze.py mult --raw p1_mult_raw.csv --out-prefix p1_mult

grid : per collision rate c
       - mean return of every model
       - paired differences (n = eval seeds): DDT_k - DT, DDT_k - uniform, DDT_k - mix_l0.9, DT - uniform
         with SEM, t, p (two-sided one-sample t on the paired diff), win rate
       - the same diff divided by (greedy - uniform) so that c values are comparable
       - figure: paired diff vs c (95% CI) for the main contrasts
mult : return vs target-RTG multiplier for each model at each c (paired over eval seeds),
       and the slope-like quantity  ret(max mult) - ret(min mult)  with its paired t.
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    d = pd.read_csv(path)
    if "mult" not in d:
        d["mult"] = 1.0
    if "base" not in d:
        d["base"] = d["model"]
    if "train_seed" in d:      # average over train seeds per eval seed (no pseudo-replication)
        d = (d.groupby(["collision", "model", "base", "mult", "seed"], as_index=False)
               .agg(ret=("ret", "mean"), max_action=("max_action", "mean")))
    return d


def paired(a, b):
    d = (a - b).dropna()
    t, p = stats.ttest_1samp(d, 0.0)
    return dict(diff=d.mean(), sem=d.std(ddof=1) / np.sqrt(len(d)), t=t, p=p,
                win=(d > 0).mean(), n=len(d))


def grid(args):
    raw = load(args.raw)
    if args.baselines:
        raw = pd.concat([raw, load(args.baselines)], ignore_index=True)
    raw = raw[raw.mult == 1.0]
    P = raw.pivot_table(index=["collision", "seed"], columns="model", values="ret")
    cols = sorted(P.index.get_level_values(0).unique())

    means = P.groupby(level=0).mean().round(2)
    means.to_csv(f"{args.out_prefix}_means.csv")
    print("\nMean return by collision rate\n", means.to_string())

    ddts = sorted([m for m in P.columns if m.startswith("DDT_k")], key=lambda s: int(s.split("_k")[1]))
    contrasts = [(m, "DT") for m in ddts]
    for base in ("uniform", "mix_l0.9", "greedy"):
        if base in P:
            contrasts += [(m, base) for m in ["DT"] + ddts]
    rows = []
    for c in cols:
        x = P.xs(c, level=0)
        gap = (x["greedy"] - x["uniform"]).mean() if {"greedy", "uniform"} <= set(x.columns) else np.nan
        for m, base in contrasts:
            if m not in x or base not in x:
                continue
            r = paired(x[m], x[base])
            rows.append(dict(collision=c, model=m, base=base, **r, greedy_minus_uniform=gap,
                             diff_over_gap=r["diff"] / gap if gap and abs(gap) > 1e-9 else np.nan))
    res = pd.DataFrame(rows)
    res.to_csv(f"{args.out_prefix}_paired.csv", index=False)

    show = res[(res.base == "DT") & res.model.isin(["DDT_k1", "DDT_k5"])]
    print("\nDDT - DT (paired)\n", show[["collision", "model", "diff", "sem", "t", "p", "win"]].round(3).to_string(index=False))
    show = res[(res.base.isin(["uniform", "mix_l0.9"])) & res.model.isin(["DT", "DDT_k1"])]
    print("\nvs uniform / mix_l0.9 (paired)\n",
          show[["collision", "model", "base", "diff", "sem", "t", "p", "win"]].round(3).to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    spec = [("DDT_k1", "DT"), ("DDT_k5", "DT"), ("DDT_k20", "DT")]
    for m, b in spec:
        s = res[(res.model == m) & (res.base == b)].set_index("collision").reindex(cols)
        ax[0].errorbar(cols, s["diff"], yerr=1.96 * s["sem"], marker="o", capsize=3, label=f"{m} - {b}")
    ax[0].set_title("Paired diff vs DT (95% CI)")
    for m, b in [("DDT_k1", "uniform"), ("DT", "uniform"), ("DDT_k1", "mix_l0.9"), ("DDT_k1", "greedy")]:
        s = res[(res.model == m) & (res.base == b)].set_index("collision").reindex(cols)
        if s["diff"].notna().any():
            ax[1].errorbar(cols, s["diff"], yerr=1.96 * s["sem"], marker="o", capsize=3, label=f"{m} - {b}")
    ax[1].set_title("Paired diff vs reference policies (95% CI)")
    s = res[(res.model == "DDT_k1") & (res.base == "DT")].set_index("collision").reindex(cols)
    ax[2].plot(cols, s["diff_over_gap"], marker="o", label="(DDT_k1 - DT) / (greedy - uniform)")
    ax[2].set_title("Diff normalised by (greedy - uniform)")
    for a in ax:
        a.axhline(0, color="k", lw=.8)
        a.axvline(0.2, color="gray", ls="--", lw=.8)          # training-data collision rate
        a.set_xlabel("evaluation collision rate c  (dashed: training data c=0.2)")
        a.grid(alpha=.4); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{args.out_prefix}.png", dpi=200); plt.close()
    print(f"\nsaved {args.out_prefix}_means.csv / _paired.csv / .png")


def mult(args):
    raw = load(args.raw)
    raw = raw[~raw.base.isin(["greedy", "random"])]
    mults = sorted(raw.mult.unique())
    lo, hi = mults[0], mults[-1]
    rows = []
    for (c, b), g in raw.groupby(["collision", "base"]):
        P = g.pivot_table(index="seed", columns="mult", values="ret")
        for mu in mults:
            rows.append(dict(collision=c, model=b, mult=mu, mean=P[mu].mean(),
                             sem=P[mu].std(ddof=1) / np.sqrt(len(P))))
        r = paired(P[hi], P[lo])
        rows.append(dict(collision=c, model=b, mult=f"x{hi:g}-x{lo:g}", mean=r["diff"], sem=r["sem"],
                         t=r["t"], p=r["p"]))
    res = pd.DataFrame(rows)
    res.to_csv(f"{args.out_prefix}_mult.csv", index=False)
    print(res.round(3).to_string(index=False))

    cs = sorted(raw.collision.unique())
    fig, ax = plt.subplots(1, len(cs), figsize=(5 * len(cs), 4), squeeze=False)
    for i, c in enumerate(cs):
        for b in sorted(raw.base.unique()):
            s = res[(res.collision == c) & (res.model == b) & res["mult"].map(lambda v: not isinstance(v, str))]
            ax[0, i].errorbar(s["mult"].astype(float), s["mean"], yerr=1.96 * s["sem"], marker="o", capsize=3, label=b)
        ax[0, i].set_title(f"c = {c:g}"); ax[0, i].set_xlabel("target RTG multiplier"); ax[0, i].grid(alpha=.4)
    ax[0, 0].set_ylabel("return"); ax[0, 0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{args.out_prefix}_mult.png", dpi=200); plt.close()
    print(f"\nsaved {args.out_prefix}_mult.csv / _mult.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid"); g.add_argument("--raw", required=True)
    g.add_argument("--baselines", default=None); g.add_argument("--out-prefix", default="p1_grid")
    m = sub.add_parser("mult"); m.add_argument("--raw", required=True); m.add_argument("--out-prefix", default="p1_mult")
    a = p.parse_args()
    grid(a) if a.cmd == "grid" else mult(a)
