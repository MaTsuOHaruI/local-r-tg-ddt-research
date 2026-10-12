"""
実験C/D: RTG_k は「行動の良し悪し」についてどれだけ情報を持つか(データ側からの検証)

検証(c)「局所化はノイズを減らしつつ信号を保つ」と、対照条件(b)の比較用です。
学習済みモデルは不要。ロールアウトデータ(または既存データセットのnpz)だけで動きます。

各RTG変種 X (global / local_k / lagged_k / random_k / expected_k) について:
    var_total   = Var(X)
    R2_s        = X を状態 s だけで予測した交差検証R^2        (episode単位で分割)
    R2_sa       = X を (s, a) で予測した交差検証R^2
    dR2         = R2_sa - R2_s                               … 行動a が X を説明する増分
    snr_action  = dR2 / (1 - R2_sa)                          … 行動由来の信号 / 残りのノイズ
  dR2・snr_action が大きいRTGほど、DTが「行動→リターン」の対応を学びやすい。
  実験Bの SNR_k・rank_acc_k と同じ向きに動くかを確認してください(別手法での相互確認)。

追加(D): Corr(RTG_k(t), 近い将来のk'-step報酬) の表。
  global RTG が直近の報酬をどれだけ反映しているか(遠い将来にどれだけ支配されているか)が分かる。

注意:
  * データの行動方策(ランダム/エキスパート混合)が結果に影響します。学習に使う実データは
    --data で渡してください(npz: obs(E,T,ds), act(E,T,da), rew(E,T))。
  * lagged/random は「局所性が本質か」を調べる対照です。局所RTGだけが高い dR2 を示し、
    lagged/random が低ければ、効果は窓の位置(直近の報酬を含むこと)に由来すると言えます。

使い方:
  python exp_C_rtg_signal.py --collisions 0.0 0.3 0.6 0.9 --episodes 300 --out out_C
  python exp_C_rtg_signal.py --data my_dataset.npz --out out_C_real
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import env_hooks as hk
import rtg_variants as rv

K_GRID_DEFAULT = [1, 2, 3, 5, 10, 20, 50]


def collect_dataset(collision, n_episodes, policy_name, seed):
    pol = hk.POLICIES[policy_name]
    rng = np.random.default_rng(seed)
    O, A, Rw = [], [], []
    for e in range(n_episodes):
        env = hk.make_env(collision, seed * 100003 + e)
        obs = env.reset(seed=seed * 100003 + e)[0]
        o, a, r = [], [], []
        for _ in range(hk.EPISODE_LENGTH):
            act = pol(obs, rng)
            o.append(np.asarray(obs, float)); a.append(act)
            obs, rew, term, trunc, _ = env.step(act)
            r.append(rew)
            if term or trunc:
                break
        O.append(o); A.append(a); Rw.append(r)
    T = min(len(x) for x in Rw)
    return (np.array([x[:T] for x in O]), np.array([x[:T] for x in A]), np.array([x[:T] for x in Rw]))


def _poly2(X):
    n, d = X.shape
    cols = [np.ones((n, 1)), X]
    for i in range(d):
        for j in range(i, d):
            cols.append((X[:, i] * X[:, j])[:, None])
    return np.hstack(cols)


def _cv_r2(X, y, groups, model="poly", n_splits=5, seed=0):
    """episode単位の交差検証R^2。model: 'poly'(2次特徴のリッジ, 高速) / 'gbm'(sklearn, 最終版向け)"""
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(ug)
    folds = np.array_split(ug, n_splits)
    pred = np.empty_like(y, dtype=float)
    for f in folds:
        te = np.isin(groups, f)
        tr = ~te
        if model == "gbm":
            from sklearn.ensemble import HistGradientBoostingRegressor
            m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=seed)
            m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
        else:
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            Ptr, Pte = _poly2((X[tr] - mu) / sd), _poly2((X[te] - mu) / sd)
            lam = 1e-2 * len(Ptr)
            w = np.linalg.solve(Ptr.T @ Ptr + lam * np.eye(Ptr.shape[1]), Ptr.T @ y[tr])
            pred[te] = Pte @ w
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def analyze(obs, act, rew, k_grid, lag=None, r_hat=None, model="poly", seed=0, variants=("global", "local", "lagged", "random")):
    E, T = rew.shape
    rng = np.random.default_rng(seed)
    per_ep = [rv.build_variants(rew[e], k_grid, lag=lag, rng=rng,
                                r_hat=None if r_hat is None else r_hat[e],
                                include=variants + (("expected",) if r_hat is not None else ()))
              for e in range(E)]
    names = list(per_ep[0].keys())
    S = obs.reshape(E * T, -1)
    SA = np.hstack([S, act.reshape(E * T, -1)])
    t_idx = np.tile(np.arange(T), E)
    S_t = np.hstack([S, t_idx[:, None]])           # 時刻も状態側に入れる(残り時間の影響を除く)
    SA_t = np.hstack([SA, t_idx[:, None]])
    groups = np.repeat(np.arange(E), T)
    rows = []
    for nm in names:
        y = np.concatenate([d[nm] for d in per_ep])
        r2s = _cv_r2(S_t, y, groups, model, seed=seed)
        r2sa = _cv_r2(SA_t, y, groups, model, seed=seed)
        d = r2sa - r2s
        rows.append(dict(variant=nm, var_total=float(y.var()), R2_s=r2s, R2_sa=r2sa, dR2=d,
                         snr_action=d / (1 - r2sa) if r2sa < 1 else np.nan))
    return pd.DataFrame(rows)


def corr_table(rew, k_grid, near=(1, 5, 10)):
    """Corr(RTG_k(t), 直近 k'-step 報酬(t..t+k'-1))。global は k='global'。"""
    E, T = rew.shape
    out = []
    for kind in ["global"] + list(k_grid):
        x = np.concatenate([rv.rtg_global(rew[e]) if kind == "global" else rv.rtg_local(rew[e], kind) for e in range(E)])
        for kp in near:
            y = np.concatenate([rv.rtg_local(rew[e], kp) for e in range(E)])
            out.append(dict(rtg="global" if kind == "global" else f"local_k{kind}", near_k=kp, corr=float(np.corrcoef(x, y)[0, 1])))
    return pd.DataFrame(out)


def make_figure(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for c, g in df.groupby("collision"):
        loc = g[g.variant.str.startswith("local_k")].copy()
        loc["k"] = loc.variant.str.extract(r"k(\d+)").astype(int)
        loc = loc.sort_values("k")
        ax[0].plot(loc.k, loc.dR2, marker="o", label=f"c={c}")
        ax[1].plot(loc.k, loc.snr_action, marker="o", label=f"c={c}")
    ax[0].set(xscale="log", xlabel="k", ylabel="ΔR² (action | state)", title="local RTG: action-explained variance")
    ax[1].set(xscale="log", xlabel="k", ylabel="ΔR² / (1-R²_sa)", title="local RTG: action signal / residual noise")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collisions", type=float, nargs="+", default=[0.0, 0.3, 0.6, 0.9])
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--k-grid", type=int, nargs="+", default=K_GRID_DEFAULT)
    ap.add_argument("--policy", default="random")
    ap.add_argument("--model", default="poly", choices=["poly", "gbm"])
    ap.add_argument("--data", default=None, help="npz(obs,act,rew)。指定時は環境を使わず、そのデータで解析")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out_C")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tabs, corrs = [], []
    if args.data:
        z = np.load(args.data)
        jobs = [(float("nan"), z["obs"], z["act"], z["rew"])]
    else:
        jobs = []
        for c in args.collisions:
            o, a, r = collect_dataset(c, args.episodes, args.policy, args.seed)
            jobs.append((c, o, a, r))
    for c, o, a, r in jobs:
        print(f"collision={c}: episodes={r.shape[0]}, T={r.shape[1]}")
        df = analyze(o, a, r, args.k_grid, model=args.model, seed=args.seed)
        df.insert(0, "collision", c)
        ct = corr_table(r, args.k_grid); ct.insert(0, "collision", c)
        tabs.append(df); corrs.append(ct)
    df, ct = pd.concat(tabs), pd.concat(corrs)
    df.to_csv(os.path.join(args.out, "C_rtg_signal.csv"), index=False)
    ct.to_csv(os.path.join(args.out, "D_rtg_vs_near_reward.csv"), index=False)
    make_figure(df, os.path.join(args.out, "C_rtg_signal.png"))
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
