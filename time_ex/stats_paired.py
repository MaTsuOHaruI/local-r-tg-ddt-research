"""
seed単位の生結果(raw_results.csv)から、主張(a)(e)(f)(g)を「統計的に」示すための集計。

入力CSV(列名は引数で変更可): collision, model, seed, return  [, scale]
  model 名は 'DT', 'DDT_k1', 'DDT_k5', 'random', 'greedy' など。DDTの k は '_k<数字>' から読み取る。

出力:
  paired_table   : 各(collision, model)のDTとのpaired差・SEM・t・95%CI・効果量dz・Wilcoxon p・Holm補正p
  dose_response  : (a) kが小さいほど改善が増えるか。seedごとに gain を log(k) に回帰した傾きの検定
  normalized_gain: (e) 改善を (DT - random) で正規化(headroomの違いを除く)。seedのブートストラップCI
  response_slope : (f) 目標スケール(または分位点)に対する性能の傾き(seedごとの傾きの平均とCI)
  power          : 差を検出するのに必要なseed数 / 現在のseed数での最小検出差

使い方:
  python stats_paired.py raw_results.csv --ref DT --out out_stats
  python stats_paired.py raw_scale_results.csv --scale-col scale --ref DT --out out_stats
"""
from __future__ import annotations
import argparse, os, re
import numpy as np
import pandas as pd
from scipy import stats


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def _wide(df, collision, seed, model, value):
    return df.pivot_table(index=[collision, seed], columns=model, values=value, aggfunc="mean")


def paired_table(df, ref="DT", collision="collision", seed="seed", model="model", value="return", alpha=0.05):
    w = _wide(df, collision, seed, model, value)
    rows = []
    for c, g in w.groupby(level=0):
        for m in g.columns:
            if m == ref or ref not in g.columns:
                continue
            d = (g[m] - g[ref]).dropna()
            n = len(d)
            if n < 3:
                continue
            mean, sd = d.mean(), d.std(ddof=1)
            sem = sd / np.sqrt(n)
            t = mean / sem if sem > 0 else np.nan
            p_t = 2 * stats.t.sf(abs(t), n - 1) if sem > 0 else np.nan
            try:
                p_w = stats.wilcoxon(d).pvalue if np.any(d != 0) else np.nan
            except ValueError:
                p_w = np.nan
            h = stats.t.ppf(1 - alpha / 2, n - 1) * sem
            rows.append(dict(collision=c, model=m, n=n, mean_diff=mean, sem=sem, t=t, p_t=p_t, p_wilcoxon=p_w,
                             ci_low=mean - h, ci_high=mean + h, dz=mean / sd if sd > 0 else np.nan))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_t"].fillna(1.0).values)
        out["significant_holm"] = out["p_holm"] < alpha
    return out


def parse_k(name):
    m = re.search(r"_k(\d+)$", str(name))
    return int(m.group(1)) if m else None


def dose_response(df, ref="DT", collision="collision", seed="seed", model="model", value="return", dt_as_k=None):
    """kが小さいほどgain(=DDT_k - DT)が大きいか。
    seedごとに gain を log(k) へ回帰 → 傾きの平均を t検定。負なら『小kほど改善』。
    dt_as_k を与えると(例: 100)、DT自身を k=dt_as_k・gain=0 の点として回帰に含める
    (『k→大でDTに収束する』という予測の検証になる)。"""
    w = _wide(df, collision, seed, model, value)
    ks = {m: parse_k(m) for m in w.columns if parse_k(m) is not None}
    rows = []
    for c, g in w.groupby(level=0):
        cols = list(ks)
        gains = g[cols].sub(g[ref], axis=0).dropna()
        x = np.log([ks[m] for m in cols])
        if dt_as_k is not None:
            x = np.append(x, np.log(dt_as_k))
            gains = gains.assign(_DT=0.0)
        slopes, rhos = [], []
        for _, row in gains.iterrows():
            y = row.values
            slopes.append(np.polyfit(x, y, 1)[0])
            rhos.append(stats.spearmanr(x, y)[0])
        slopes, rhos = np.array(slopes), np.array(rhos)
        n = len(slopes)
        if n < 3:
            continue
        t, p = stats.ttest_1samp(slopes, 0.0)
        rows.append(dict(collision=c, n=n, mean_slope_per_logk=slopes.mean(), sem=slopes.std(ddof=1) / np.sqrt(n),
                         t=t, p=p, mean_spearman=np.nanmean(rhos),
                         p_wilcoxon=stats.wilcoxon(slopes).pvalue if np.any(slopes != 0) else np.nan))
    return pd.DataFrame(rows)


def normalized_gain(df, model, ref="DT", floor="random", collision="collision", seed="seed", name="model",
                    value="return", n_boot=2000, rng_seed=0):
    """gain / (ref - floor) をseedのブートストラップでCI付きで出す。
    分母(ref-floor)のCIが0を含む条件では不安定なので warn 列で知らせる。"""
    w = _wide(df, collision, seed, name, value)
    rng = np.random.default_rng(rng_seed)
    rows = []
    for c, g in w.groupby(level=0):
        if any(x not in g.columns for x in (model, ref, floor)):
            continue
        g = g[[model, ref, floor]].dropna()
        n = len(g)
        gain, head = (g[model] - g[ref]).values, (g[ref] - g[floor]).values
        point = gain.mean() / head.mean() if head.mean() != 0 else np.nan
        bs, hb = [], []
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            hm = head[i].mean()
            hb.append(hm)
            bs.append(gain[i].mean() / hm if hm != 0 else np.nan)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        hlo, hhi = np.percentile(hb, [2.5, 97.5])
        rows.append(dict(collision=c, model=model, n=n, gain=gain.mean(), headroom=head.mean(),
                         norm_gain=point, ci_low=lo, ci_high=hi, warn_headroom_ci_includes_0=bool(hlo <= 0 <= hhi)))
    return pd.DataFrame(rows)


def response_slope(df, scale_col="scale", collision="collision", seed="seed", model="model", value="return", ref="DT"):
    """目標スケール(または目標分位点)に対する性能の傾き。seedごとにOLS → 平均とCI。
    ref との差(paired)も出す。『RTGに応答する』ことの指標(f)。
    ※ kごとに目標の動かし幅が違う問題を避けるため、scale_col は『その手法のRTG分布の分位点』で揃えるのが望ましい。"""
    rows, per_seed = [], {}
    for (c, m, s), g in df.groupby([collision, model, seed]):
        if g[scale_col].nunique() < 2:
            continue
        per_seed[(c, m, s)] = np.polyfit(g[scale_col].values, g[value].values, 1)[0]
    ser = pd.Series(per_seed)
    ser.index.names = ["collision", "model", "seed"]
    for (c, m), g in ser.groupby(level=[0, 1]):
        v = g.values
        n = len(v)
        sem = v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        h = stats.t.ppf(0.975, n - 1) * sem if n > 1 else np.nan
        row = dict(collision=c, model=m, n=n, slope=v.mean(), sem=sem, ci_low=v.mean() - h, ci_high=v.mean() + h)
        if m != ref and (c, ref) in ser.groupby(level=[0, 1]).groups:
            r = ser.xs((c, ref), level=[0, 1])
            mm = ser.xs((c, m), level=[0, 1])
            dd = (mm - r).dropna()
            if len(dd) > 2:
                row.update(diff_vs_ref=dd.mean(), diff_t=stats.ttest_1samp(dd, 0.0)[0], diff_p=stats.ttest_1samp(dd, 0.0)[1])
        rows.append(row)
    return pd.DataFrame(rows)


def required_n(sd_diff, effect, alpha=0.05, power=0.8):
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil((z * sd_diff / effect) ** 2))


def min_detectable_effect(sd_diff, n, alpha=0.05, power=0.8):
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * sd_diff / np.sqrt(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--ref", default="DT")
    ap.add_argument("--floor", default="random")
    ap.add_argument("--collision-col", default="collision")
    ap.add_argument("--seed-col", default="seed")
    ap.add_argument("--model-col", default="model")
    ap.add_argument("--value-col", default="return")
    ap.add_argument("--scale-col", default=None)
    ap.add_argument("--dt-as-k", type=int, default=None, help="DTをk=この値の点として用量反応に含める(例: 100)")
    ap.add_argument("--out", default="out_stats")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.csv)
    kw = dict(collision=a.collision_col, seed=a.seed_col, value=a.value_col)

    if a.scale_col:
        rs = response_slope(df, a.scale_col, a.collision_col, a.seed_col, a.model_col, a.value_col, a.ref)
        rs.to_csv(os.path.join(a.out, "f_response_slope.csv"), index=False)
        print(rs.round(4).to_string(index=False))
        return

    pt = paired_table(df, a.ref, model=a.model_col, **kw)
    pt.to_csv(os.path.join(a.out, "a_paired_table.csv"), index=False)
    dr = dose_response(df, a.ref, model=a.model_col, dt_as_k=a.dt_as_k, **kw)
    dr.to_csv(os.path.join(a.out, "a_dose_response.csv"), index=False)
    ng = pd.concat([normalized_gain(df, m, a.ref, a.floor, name=a.model_col, **kw)
                    for m in df[a.model_col].unique() if parse_k(m) is not None], ignore_index=True)
    ng.to_csv(os.path.join(a.out, "e_normalized_gain.csv"), index=False)
    pw = pt.assign(sd_diff=pt["sem"] * np.sqrt(pt["n"]))
    pw["mde_80pct_power"] = [min_detectable_effect(s, n) for s, n in zip(pw["sd_diff"], pw["n"])]
    pw[["collision", "model", "n", "mean_diff", "sd_diff", "mde_80pct_power"]].to_csv(os.path.join(a.out, "g_power.csv"), index=False)
    for nm, t in [("paired", pt), ("dose-response", dr), ("normalized gain", ng)]:
        print(f"\n== {nm} ==")
        print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
