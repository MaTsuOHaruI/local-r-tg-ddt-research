"""
環境との接続部。実環境(WifiEnv)に合わせて編集するのは「USER SECTION」だけです。

共通乱数(CRN)の考え方:
  「行動aと行動a'の違いだけ」を測るには、外生乱数(CSIノイズ・到着・衝突)を
  両方の分岐で同一にする必要があります。そのため
    1) 分岐の直前に環境をコピー(snapshot)し、
    2) 乱数生成器を同じseedで再設定(reseed)し、
    3) 外生乱数が行動に依存しない回数・順序で引かれること
  を前提にしています。3)は check_crn() で必ず確認してください。
"""
from __future__ import annotations
import copy
import random
import numpy as np

# =====================  USER SECTION  =====================
def make_env(collision: float, seed: int):
    from wifi_env_data_collect_v2 import WifiEnv

    env = WifiEnv(
        collision_rate=collision
    )
    env.reset(seed=seed)

    return env


ACTION_DIM = 3
EPISODE_LENGTH = 100
# ===========================================================


def _unwrapped(env):
    return getattr(env, "unwrapped", env)


def snapshot(env):
    """環境の完全コピー(乱数状態を含む)。"""
    return copy.deepcopy(env)


def reseed(env, seed: int) -> None:
    e = _unwrapped(env)

    if hasattr(e, "_rng") and isinstance(e._rng, np.random.Generator):
        e._rng = np.random.default_rng(seed)

    elif hasattr(e, "rng") and isinstance(e.rng, np.random.Generator):
        e.rng = np.random.default_rng(seed)

    elif hasattr(e, "np_random"):
        try:
            e.np_random = np.random.default_rng(seed)
        except Exception:
            e._np_random = np.random.default_rng(seed)

    else:
        np.random.seed(seed % (2**32))
        random.seed(seed)


def rng_fingerprint(env) -> float:
    """乱数生成器の「次の出力」を覗く(環境を壊さない)。CRN検査用。"""
    e = _unwrapped(env)
    for name in ("_rtg", "rng", "np_random", "_np_random"):
        g = getattr(e, name, None)
        if isinstance(g, np.random.Generator):
            return float(copy.deepcopy(g).random())
    # 全域のnumpy乱数を使う環境: 内部位置と状態の一部を指紋にする(消費量が違えば値が変わる)
    st = np.random.get_state()
    return float(int(st[2]) * 1000003 + int(st[1][:16].sum() % 1000003))


def dirichlet_action(rng: np.random.Generator, alpha: float = 1.0) -> np.ndarray:
    return rng.dirichlet(np.full(ACTION_DIM, alpha))


def random_policy(obs, rng: np.random.Generator) -> np.ndarray:
    return dirichlet_action(rng)


# 追加の行動方策(greedy等)はここに登録: policy(obs, rng) -> action
POLICIES = {"random": random_policy}


def rollout_open_loop(env_snap, actions: np.ndarray, noise_seed: int) -> np.ndarray:
    """env_snap をコピーし、乱数を noise_seed で再設定して actions を順に実行。報酬列(H,)を返す。
    エピソードが先に終わった場合は残りを NaN にする。"""
    env = snapshot(env_snap)
    reseed(env, noise_seed)
    H = len(actions)
    rew = np.full(H, np.nan)
    for j in range(H):
        _, r, term, trunc, _ = env.step(actions[j])
        rew[j] = r
        if term or trunc:
            break
    return rew


def check_crn(make_env_fn=make_env, collision: float = 0.3, H: int = 20, n_trials: int = 5, seed: int = 0) -> dict:
    """CRNの前提を検査する。
    (1) 決定性: 同じ状態・同じseed・同じ行動列 → 完全に同じ報酬列か
    (2) 乱数の消費が行動に依存しない: 異なる行動列を実行した後の乱数生成器の「次の出力」が一致するか
    どちらかが False なら、行動比較(実験B)の結果は信用できません。"""
    rng = np.random.default_rng(seed)
    ok_det, ok_rng = True, True
    for i in range(n_trials):
        env = make_env_fn(collision, seed + i)
        for _ in range(int(rng.integers(0, 30))):
            env.step(dirichlet_action(rng))
        snap = snapshot(env)
        acts_a = np.stack([dirichlet_action(rng) for _ in range(H)])
        acts_b = np.stack([dirichlet_action(rng) for _ in range(H)])
        r1 = rollout_open_loop(snap, acts_a, 123)
        r2 = rollout_open_loop(snap, acts_a, 123)
        ok_det &= bool(np.array_equal(r1, r2, equal_nan=True))
        fps = []
        for acts in (acts_a, acts_b):
            e = snapshot(snap)
            reseed(e, 123)
            for j in range(H):
                e.step(acts[j])
            fps.append(rng_fingerprint(e))
        ok_rng &= bool(abs(fps[0] - fps[1]) < 1e-12)
    return {"deterministic_given_seed": ok_det, "rng_consumption_independent_of_action": ok_rng}
