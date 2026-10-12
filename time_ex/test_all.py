"""動作確認テスト: python test_all.py  (実環境なしで、コードの正しさと前提検査を確認)"""
import numpy as np, pandas as pd
import rtg_variants as rv
import env_hooks as hk
import stats_paired as sp


def test_rtg():
    r = np.arange(1.0, 11.0)
    assert np.allclose(rv.rtg_local(r, 10), rv.rtg_global(r))
    assert np.allclose(rv.rtg_local(r, 100), rv.rtg_global(r))
    assert np.allclose(rv.rtg_local(r, 1), r)
    assert np.allclose(rv.rtg_local(r, 3)[:2], [6, 9]) and rv.rtg_local(r, 3)[-1] == 10
    lag = rv.rtg_lagged(r, 2, 3)               # t=0: r[3]+r[4]=4+5
    assert lag[0] == 9 and lag[-1] == 0 and lag[-4] == 10   # t=6: r[9]
    rw = rv.rtg_random_window(r, 4, np.random.default_rng(0))
    assert rw.shape == r.shape and np.all(rw >= 1 + 2 + 3 + 4) and np.all(rw <= 7 + 8 + 9 + 10)
    z, m, s = rv.standardize(rv.rtg_global(r)); assert abs(z.mean()) < 1e-9 and abs(z.std() - 1) < 1e-9
    v = rv.build_variants(r, [1, 3], r_hat=r * 0.5)
    assert {"global", "local_k1", "lagged_k3", "random_k3", "expected_k3"} <= set(v)


def test_crn():
    assert all(hk.check_crn().values())

    class Broken:                                   # 行動に応じて乱数の消費量が変わる環境
        def __init__(s, seed=0): s.rng = np.random.default_rng(seed); s.t = 0
        def reset(s, seed=None): return np.zeros(2), {}
        def step(s, a):
            if a[0] > 0.33: s.rng.random()          # ← 行動依存の消費
            s.t += 1
            return np.zeros(2), float(s.rng.random()), False, False, {}
    res = hk.check_crn(make_env_fn=lambda c, sd: Broken(sd), H=10)
    assert not res["rng_consumption_independent_of_action"]


def test_stats_recovers_known_effects():
    # 提示された実データの論点: collision=0.3 の paired差(DDT_k1=+6.6, sem 0.91, n=50) を再現できるか
    rng = np.random.default_rng(1)
    n = 50
    rows = []
    base = rng.normal(803, 30, n)
    for sd, (mdl, eff, sem) in enumerate([("DT", 0, 0), ("DDT_k1", 6.6, 0.91), ("DDT_k5", 3.1, 0.54),
                                         ("DDT_k20", 1.0, 0.23), ("DDT_k50", 0.2, 0.13), ("random", -6.7, 3.4)]):
        d = rng.normal(eff, sem * np.sqrt(n), n) if sem else np.zeros(n)
        for s in range(n):
            rows.append(dict(collision=0.3, model=mdl, seed=s, **{"return": base[s] + d[s]}))
    df = pd.DataFrame(rows)
    pt = sp.paired_table(df).set_index("model")
    assert pt.loc["DDT_k1", "t"] > 4 and pt.loc["DDT_k1", "significant_holm"]
    assert abs(pt.loc["DDT_k1", "mean_diff"] - 6.6) < 1.5
    dr = sp.dose_response(df)
    assert dr.loc[0, "mean_slope_per_logk"] < 0 and dr.loc[0, "p"] < 0.01      # 小kほど改善
    dr2 = sp.dose_response(df, dt_as_k=100)
    assert dr2.loc[0, "p"] < 0.01
    ng = sp.normalized_gain(df, "DDT_k1")
    assert ng.loc[0, "norm_gain"] > 0
    sd = 1.917 * np.sqrt(50)   # collision=0.9 のDDT_k1のpaired差のsd
    assert sp.required_n(sd, 0.8) > 1000 and abs(sp.min_detectable_effect(sd, 50) - 5.4) < 0.3


def test_response_slope():
    rng = np.random.default_rng(2)
    rows = []
    for m, slope in [("DT", 1.0), ("DDT_k1", 25.0)]:
        for s in range(30):
            for sc in (0.5, 1.0, 1.5):
                rows.append(dict(collision=0.3, model=m, seed=s, scale=sc, **{"return": 800 + slope * sc + rng.normal(0, 3)}))
    rs = sp.response_slope(pd.DataFrame(rows)).set_index("model")
    assert rs.loc["DDT_k1", "slope"] > 20 and rs.loc["DDT_k1", "diff_p"] < 1e-6 and rs.loc["DT", "slope"] < 5


def test_expB_runs_and_has_sane_values():
    import exp_B_action_horizon as eb
    k_df, lag_df, summ = eb.analyze_collision(0.3, n_states=12, R=6, H=20, k_grid=[1, 5, 20],
                                              policy_name="random", seed=0, verbose=False)
    assert len(k_df) == 3 and len(lag_df) == 20
    assert np.all(np.isfinite(k_df["cum_eff"])) and np.all((k_df["rank_acc"] >= 0) & (k_df["rank_acc"] <= 1))
    assert lag_df["eff"].iloc[0] >= lag_df["eff"].iloc[-1]          # mock: 行動効果は時間とともに減衰


def test_expC_runs():
    import exp_C_rtg_signal as ec
    o, a, r = ec.collect_dataset(0.3, 60, "random", 0)
    df = ec.analyze(o, a, r, [1, 10], seed=0)
    loc = df.set_index("variant")
    assert loc.loc["local_k1", "dR2"] > loc.loc["random_k1", "dR2"]  # 局所RTGは行動情報を持つ、ランダム窓は持たない


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
