# dataset_fixed.py  (v2)
"""
Offline dataset for DT (mode="global") and DDT (mode="local_k").

Changes
-------
[D-1] Local RTG follows the research definition
          LRTG_t^(k) = sum_{i=t}^{min(t+k,T)-1} r_i        (k reward terms)
      (the old docstring wrote the sum up to t+k inclusive; the code always used
      k terms, which is the definition used in the research).
      `tail="truncate"` keeps the shrinking window at the episode end,
      `tail="rescale"` multiplies it by k/(window length).  Vectorised via cumsum.
[D-2] RTG is NORMALISED by (k_used * mean_step_reward) so that the network input is
      O(1) for every k (k=1 -> ~9, k=50 -> ~450, global -> ~900 would otherwise differ
      by 2 orders of magnitude).  The same constant is stored in the checkpoint.
[D-3] Window sampling "end_aligned" (default): sample the END step e uniformly and use
      steps [max(0, e-C+1), e].  This is exactly what evaluation sees (the last C steps
      ending at the current step); the old "uniform start" sampled t=0 only 1 % of the
      time, so early-episode behaviour was under-trained.
[D-4] __len__ = num_episodes * windows_per_episode (the old dataset gave ONE random
      window per episode per epoch: ~15 gradient steps/epoch for 1000 episodes).
[D-5] Dedicated RNG (seed argument), DataLoader-worker safe.
"""
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info


def local_rtg_array(rewards, k, tail="truncate"):
    """rewards [N,T] -> Local RTG [N,T] (float64).  k=None or k>=T -> ordinary RTG."""
    assert tail in ("truncate", "rescale")
    r = np.asarray(rewards, dtype=np.float64)
    N, T = r.shape
    if k is None or k >= T:
        k = T
    c = np.concatenate([np.zeros((N, 1)), np.cumsum(r, axis=1)], axis=1)
    t = np.arange(T)
    end = np.minimum(t + k, T)
    out = c[:, end] - c[:, t]
    if tail == "rescale":
        out = out * (k / (end - t))[None, :]
    return out


class WifiOfflineDataset(Dataset):
    def __init__(self, filepath, context_len=20, mode="global",
                 k_min=1, k_max=50, tail="truncate",
                 windows_per_episode=20, window_sampling="end_aligned", seed=0):
        assert mode in ("global", "local_k")
        assert window_sampling in ("end_aligned", "uniform_start")
        with open(filepath, "rb") as f:
            self.data = pickle.load(f)

        self.context_len = context_len
        self.mode = mode
        self.k_min, self.k_max = int(k_min), int(k_max)
        self.tail = tail
        self.windows_per_episode = windows_per_episode
        self.window_sampling = window_sampling
        self.seed = seed

        self.states = self.data["states"]
        self.actions = self.data["actions"]
        self.rewards = self.data["rewards"]
        self.rtgs = self.data["returns_to_go"]
        self.num_episodes, self.max_len = self.states.shape[:2]

        r = self.rewards.astype(np.float64)
        self.mean_reward = float(r.mean())
        self.csum = np.concatenate([np.zeros((self.num_episodes, 1)), np.cumsum(r, axis=1)], axis=1)

        self._rng = np.random.default_rng(seed)
        self._worker_rng = {}

    def __len__(self):
        return self.num_episodes * self.windows_per_episode

    def _get_rng(self):
        info = get_worker_info()
        if info is None:
            return self._rng
        if info.seed not in self._worker_rng:
            self._worker_rng = {info.seed: np.random.default_rng(info.seed)}
        return self._worker_rng[info.seed]

    def _local_window(self, ep, ts, k):
        T = self.max_len
        end = np.minimum(ts + k, T)
        out = self.csum[ep, end] - self.csum[ep, ts]
        if self.tail == "rescale":
            out = out * (k / (end - ts))
        return out

    def __getitem__(self, idx):
        rng = self._get_rng()
        ep = idx % self.num_episodes
        T, C = self.max_len, self.context_len

        if self.window_sampling == "end_aligned":
            e = int(rng.integers(0, T))
            t0, t1 = max(0, e - C + 1), e + 1
        else:
            t0 = int(rng.integers(0, T))
            t1 = min(t0 + C, T)
        ts = np.arange(t0, t1)
        slice_len = t1 - t0
        pad_len = C - slice_len

        s = self.states[ep, t0:t1]
        a = self.actions[ep, t0:t1]

        if self.mode == "global":
            k_used = T
            raw = self.rtgs[ep, t0:t1].astype(np.float64)
        else:
            k_used = self.k_min if self.k_min == self.k_max else int(rng.integers(self.k_min, self.k_max + 1))
            raw = self._local_window(ep, ts, k_used)

        rtg = (raw / (k_used * self.mean_reward)).reshape(-1, 1)
        timesteps = ts

        if pad_len > 0:
            s = np.concatenate([s, np.zeros((pad_len, s.shape[1]))], axis=0)
            a = np.concatenate([a, np.zeros((pad_len, a.shape[1]))], axis=0)
            rtg = np.concatenate([rtg, np.zeros((pad_len, 1))], axis=0)
            timesteps = np.concatenate([timesteps, np.zeros(pad_len, dtype=np.int64)])

        mask = np.ones(C, dtype=np.float32)
        if pad_len > 0:
            mask[-pad_len:] = 0.0

        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(rtg, dtype=torch.float32),
            torch.tensor(timesteps, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(float(k_used), dtype=torch.float32),
        )
