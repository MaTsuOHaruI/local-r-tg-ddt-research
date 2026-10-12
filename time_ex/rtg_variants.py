"""
RTG(Return-To-Go)の変種を作る関数群。学習用データセット作成側(dataset_fixed.py等)から
そのまま呼べるよう、1エピソードの報酬配列 r (T,) だけを入力にしています。

検証(b)「効果の原因は時間的局所性か」の対照条件を含みます:

  global        : 通常のRTG  (t以降の全報酬の和)
  local_k       : 局所RTG    (t..t+k-1 の和; エピソード末で切り詰め)           ← 提案手法
  lagged_k      : 遅延窓     (t+lag..t+lag+k-1 の和)
                  同じ項数(=同程度の分散)で「直近の報酬を含まない」窓。
                  局所性が本質なら、これは効果が消えるはず。
  random_k      : ランダム窓 (エピソード内から無作為に選んだk個の報酬の和)
                  項数・スケールは同じで時間的対応だけを壊した対照。
  expected_k    : 期待値窓   (実報酬の代わりに E[r|s,a] の予測値で作る局所RTG)
                  「窓で切ったこと」と「ノイズを除いたこと」を分離するための条件。

注意:
  * どの変種も学習時と評価時で同じ定義を使うこと。
  * スケールの影響を除くため、学習時に standardize() で全変種を同じ平均・分散に揃える
    ことを推奨します(スケール交絡の排除)。
"""
from __future__ import annotations
import numpy as np


def _csum(r: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(r, dtype=float)])


def rtg_global(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return np.cumsum(r[::-1])[::-1]


def rtg_local(r: np.ndarray, k: int) -> np.ndarray:
    """sum_{j=t}^{min(t+k-1, T-1)} r_j"""
    return rtg_lagged(r, k, lag=0)


def rtg_lagged(r: np.ndarray, k: int, lag: int) -> np.ndarray:
    """sum_{j=t+lag}^{t+lag+k-1} r_j (エピソード末で切り詰め, 範囲外は0)"""
    r = np.asarray(r, dtype=float)
    T = len(r)
    c = _csum(r)
    t = np.arange(T)
    a = np.minimum(t + lag, T)
    b = np.minimum(t + lag + k, T)
    return c[b] - c[a]


def rtg_random_window(r: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """各tで、エピソード内の相異なるk個の時刻を無作為に選んで報酬を足す。"""
    r = np.asarray(r, dtype=float)
    T = len(r)
    k = min(k, T)
    out = np.empty(T)
    for t in range(T):
        idx = rng.choice(T, size=k, replace=False)
        out[t] = r[idx].sum()
    return out


def rtg_expected_local(r_hat: np.ndarray, k: int) -> np.ndarray:
    """r_hat[t] = E[r_t | s_t, a_t] の推定値。実報酬の代わりに使う局所RTG。"""
    return rtg_local(r_hat, k)


def standardize(x: np.ndarray, mean: float | None = None, std: float | None = None):
    """データセット全体で標準化。(標準化後の配列, mean, std) を返す。
    評価時に同じ mean/std を使うために mean,std も返します。"""
    x = np.asarray(x, dtype=float)
    m = x.mean() if mean is None else mean
    s = x.std() if std is None else std
    s = s if s > 0 else 1.0
    return (x - m) / s, m, s


def build_variants(r, k_list, lag=None, rng=None, r_hat=None, include=("global", "local", "lagged", "random", "expected")):
    """1エピソード分の全変種を dict で返す。キー例: 'global', 'local_k5', 'lagged_k5', ...
    lag を省略すると各kについて lag=k(直近k stepを丸ごと飛ばす窓)を使う。"""
    r = np.asarray(r, dtype=float)
    rng = rng if rng is not None else np.random.default_rng(0)
    out = {}
    if "global" in include:
        out["global"] = rtg_global(r)
    for k in k_list:
        if "local" in include:
            out[f"local_k{k}"] = rtg_local(r, k)
        if "lagged" in include:
            out[f"lagged_k{k}"] = rtg_lagged(r, k, lag if lag is not None else k)
        if "random" in include:
            out[f"random_k{k}"] = rtg_random_window(r, k, rng)
        if "expected" in include and r_hat is not None:
            out[f"expected_k{k}"] = rtg_expected_local(r_hat, k)
    return out
