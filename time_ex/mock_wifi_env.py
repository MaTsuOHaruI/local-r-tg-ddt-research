"""
MockWifiEnv: テスト専用の代用環境（実際のWifiEnvではありません）。

目的: このパッケージのコードが動くこと・CRN(共通乱数)の前提が守られていることを
      実環境なしで確認するためだけに使います。
      論文・発表に載せる数値は必ず実環境(WifiEnv)で出してください。

想定は研究メモの記述に合わせています:
  NUM_USERS=3, episode_length=100, state_dim=6(CSI3+queue3), action_dim=3(simplex),
  CSI: 平均回帰の確率過程, queue: 0〜50, 到着はPoisson(3), reward=総throughput,
  collision時は影響を受けたuserのthroughputが0になる。

重要な設計: 外生乱数(CSIノイズ・到着・衝突)は「行動に依存しない回数・順序」で毎step引く。
           これが共通乱数(CRN)比較の前提条件です。
"""
import numpy as np


class MockWifiEnv:
    def __init__(self, collision_prob=0.3, num_users=3, episode_length=100, seed=0,
                 arrival_lambda=3.0, csi_mean=5.5, csi_noise=0.15 * 5.5,
                 csi_revert=0.05, rate_scale=2.0, queue_max=50.0):
        self.p = float(collision_prob)
        self.U = num_users
        self.T = episode_length
        self.lam = arrival_lambda
        self.csi_mean = csi_mean
        self.csi_noise = csi_noise
        self.csi_revert = csi_revert
        self.rate_scale = rate_scale
        self.queue_max = queue_max
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.csi = None
        self.queue = None

    # --- gymnasium風API ---
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.csi = np.clip(self.rng.normal(self.csi_mean, 1.5, self.U), 1.0, 10.0)
        self.queue = np.clip(self.rng.poisson(self.lam, self.U).astype(float), 0, self.queue_max)
        return self._obs(), {}

    def _obs(self):
        return np.concatenate([self.csi, self.queue]).astype(np.float32)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), 1e-8, None)
        a = a / a.sum()
        # --- 外生乱数: 行動に依存しない回数・順序で必ず引く ---
        csi_eps = self.rng.normal(size=self.U)
        arrivals = self.rng.poisson(self.lam, self.U).astype(float)
        coll_event = self.rng.random() < self.p
        coll_mask = self.rng.random(self.U) < 0.6
        # --- ダイナミクス ---
        rate = a * self.csi * self.rate_scale
        thr = np.minimum(self.queue, rate)
        if coll_event:
            thr = np.where(coll_mask, 0.0, thr)
        reward = float(thr.sum())
        self.queue = np.clip(self.queue - thr + arrivals, 0.0, self.queue_max)
        self.csi = np.clip(
            self.csi + self.csi_revert * (self.csi_mean - self.csi) + self.csi_noise * csi_eps,
            1.0, 10.0)
        self.t += 1
        truncated = self.t >= self.T
        return self._obs(), reward, False, truncated, {}
