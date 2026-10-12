"""
実験B: 行動の効果は何step先まで残るか / どのkのRTGが「良い行動」を見分けられるか

検証(d)「最適kは行動効果の持続長で決まる」と(c)「ノイズ低減と信号保持」の直接検証です。
学習済みモデルは不要で、環境だけで動きます(共通乱数CRNを使用)。

手順(各collisionについて):
  1. 行動方策(既定: ランダム)でt0 stepだけ進めた状態を N 個作る。
  2. 各状態で、最初の行動 a と a' を2通り選び、以降の行動列 f は共通(open-loop)にする。
  3. 外生乱数を共通にして a 側/a' 側を実行 → 報酬差 Δ_j (j=0..H-1) を得る。
        eff_j     = E|Δ_j|             … lag j での「行動の効果の大きさ」
        cum_eff_k = E|Σ_{j<k} Δ_j|    … 最初の行動がk-step returnに与える効果
  4. 乱数seedだけを変えた R 本の複製から、k-step returnのノイズ
        sd_k = std_over_seeds( Σ_{j<k} r_j )   を測る。
  5. 指標
        SNR_k      = cum_eff_k / sd_k
        rank_acc_k = 独立な2本の軌跡(別々のノイズ)の k-step return を比べたとき、
                     期待値の意味で良い行動を正しく選べる確率
                     (オフラインデータ中のRTG_kが「良い行動」を見分ける能力そのもの)
  6. 実効horizon h*_eps = eff_j が eff_0 の eps 倍を下回る最小の j。

予測: 行動効果が短期なら cum_eff_k は小さいkで飽和し、sd_k は増え続けるので
      SNR_k / rank_acc_k は小さいkで最大になる。効果が長く残る環境ほど最適kは大きくなる。

使い方:
  python exp_B_action_horizon.py --collisions 0.0 0.3 0.6 0.9 --n-states 100 --R 20 --H 50 --out out_B
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import pandas as pd
import env_hooks as hk

K_GRID_DEFAULT = [1, 2, 3, 5, 10, 20, 50]


def collect_start_states(collision, n_states, t_range, policy, seed):
    rng = np.random.default_rng(seed)
    snaps = []
    for i in range(n_states):
        env = hk.make_env(collision, seed * 100003 + i)
        obs = env.reset(seed=seed * 100003 + i)[0] if hasattr(env, "reset") else None
        t0 = int(rng.integers(t_range[0], t_range[1] + 1))
        for _ in range(t0):
            obs, _, term, trunc, _ = env.step(policy(obs, rng))
            if term or trunc:
                break
        snaps.append(hk.snapshot(env))
    return snaps


def analyze_collision(collision, n_states, R, H, k_grid, policy_name, seed, verbose=True):
    policy = hk.POLICIES[policy_name]
    t_max = hk.EPISODE_LENGTH - H
    if t_max < 0:
        raise ValueError("H がエピソード長より長い")
    snaps = collect_start_states(collision, n_states, (0, t_max), policy, seed)
    rng = np.random.default_rng(seed + 1)

    abs_delta = np.zeros((n_states, H))           # |Δ_j| の noise-seed平均
    cum_eff = np.zeros((n_states, len(k_grid)))
    sd_k = np.zeros((n_states, len(k_grid)))
    acc = np.full((n_states, len(k_grid)), np.nan)
    exp_diff = np.zeros((n_states, len(k_grid)))
    t_start = time.time()

    for s, snap in enumerate(snaps):
        a = hk.dirichlet_action(rng)
        a2 = hk.dirichlet_action(rng)
        f = np.stack([hk.dirichlet_action(rng) for _ in range(H - 1)]) if H > 1 else np.zeros((0, hk.ACTION_DIM))
        acts_a = np.vstack([a[None], f])
        acts_b = np.vstack([a2[None], f])
        seeds = rng.integers(0, 2 ** 31 - 1, size=2 * R)
        ra = np.stack([hk.rollout_open_loop(snap, acts_a, int(seeds[i])) for i in range(R)])       # (R,H)
        rb_crn = np.stack([hk.rollout_open_loop(snap, acts_b, int(seeds[i])) for i in range(R)])   # 同じseed(CRN)
        rb_ind = np.stack([hk.rollout_open_loop(snap, acts_b, int(seeds[R + i])) for i in range(R)])  # 独立seed

        delta = ra - rb_crn                                  # CRN差 (R,H)
        abs_delta[s] = np.nanmean(np.abs(delta), axis=0)
        ca, cb = np.cumsum(ra, axis=1), np.cumsum(rb_crn, axis=1)
        cb_ind = np.cumsum(rb_ind, axis=1)
        for ki, k in enumerate(k_grid):
            kk = min(k, H) - 1
            d_k = ca[:, kk] - cb[:, kk]                      # CRN下の k-step return 差 (R,)
            cum_eff[s, ki] = abs(np.nanmean(d_k))            # 期待差(ノイズ平均)の大きさ
            exp_diff[s, ki] = np.nanmean(d_k)
            sd_k[s, ki] = np.nanstd(ca[:, kk], ddof=1)
            # 独立ノイズで比べた時に、期待値の意味で良い方を選べる確率
            if abs(exp_diff[s, ki]) > 1e-9:
                indep = ca[:, kk] - cb_ind[:, kk]            # 別軌跡どうしの比較
                acc[s, ki] = np.mean(np.sign(indep) == np.sign(exp_diff[s, ki]))
        if verbose and (s + 1) % max(1, n_states // 5) == 0:
            print(f"  collision={collision}: {s+1}/{n_states} states ({time.time()-t_start:.0f}s)")

    eff_j = np.nanmean(abs_delta, axis=0)
    rows = []
    for ki, k in enumerate(k_grid):
        ce = float(np.mean(cum_eff[:, ki]))
        sd = float(np.mean(sd_k[:, ki]))
        rows.append(dict(collision=collision, k=k, cum_eff=ce, sd=sd, snr=ce / sd if sd > 1e-9 else np.nan,
                         rank_acc=float(np.nanmean(acc[:, ki])), n_states=n_states, R=R))
    lag_rows = [dict(collision=collision, lag=j, eff=float(eff_j[j])) for j in range(H)]
    hstar = {}
    for eps in (0.1, 0.05):
        below = np.where(eff_j < eps * eff_j[0])[0]
        hstar[eps] = int(below[0]) if len(below) else None
    summ = dict(collision=collision, eff0=float(eff_j[0]), h_star_10pct=hstar[0.1], h_star_5pct=hstar[0.05])
    return pd.DataFrame(rows), pd.DataFrame(lag_rows), summ


def make_figure(k_df, lag_df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for c, g in lag_df.groupby("collision"):
        ax[0].plot(g["lag"], g["eff"] / g["eff"].iloc[0], label=f"c={c}")
    ax[0].set(xlabel="lag j", ylabel="E|Δ_j| / E|Δ_0|", title="action effect vs lag")
    for c, g in k_df.groupby("collision"):
        ax[1].plot(g["k"], g["snr"], marker="o", label=f"c={c}")
        ax[2].plot(g["k"], g["rank_acc"], marker="o", label=f"c={c}")
    ax[1].set(xscale="log", xlabel="k", ylabel="SNR_k", title="effect / noise of k-step return")
    ax[2].set(xscale="log", xlabel="k", ylabel="rank accuracy", title="P(RTG_k picks better action)")
    ax[2].axhline(0.5, color="gray", ls=":")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collisions", type=float, nargs="+", default=[0.0, 0.3, 0.6, 0.9])
    ap.add_argument("--n-states", type=int, default=100)
    ap.add_argument("--R", type=int, default=20, help="noise-seed replicates per branch")
    ap.add_argument("--H", type=int, default=50)
    ap.add_argument("--k-grid", type=int, nargs="+", default=K_GRID_DEFAULT)
    ap.add_argument("--policy", default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out_B")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    chk = hk.check_crn(collision=args.collisions[0])
    print("CRN check:", chk)
    if not all(chk.values()):
        raise SystemExit("CRN前提が満たされていません。env_hooks.reseed()/環境の乱数管理を確認してください。")

    ks, lags, summs = [], [], []
    for c in args.collisions:
        k_df, lag_df, summ = analyze_collision(c, args.n_states, args.R, args.H, args.k_grid, args.policy, args.seed)
        ks.append(k_df); lags.append(lag_df); summs.append(summ)
    k_df, lag_df, s_df = pd.concat(ks), pd.concat(lags), pd.DataFrame(summs)
    k_df.to_csv(os.path.join(args.out, "B_k_metrics.csv"), index=False)
    lag_df.to_csv(os.path.join(args.out, "B_lag_effect.csv"), index=False)
    s_df.to_csv(os.path.join(args.out, "B_summary.csv"), index=False)
    make_figure(k_df, lag_df, os.path.join(args.out, "B_action_horizon.png"))
    print(k_df.round(3).to_string(index=False))
    print(s_df.to_string(index=False))


if __name__ == "__main__":
    main()
