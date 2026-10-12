# wifi_env_data_collect_v2.py
"""
================================================================================
  Wi-Fi / OFDMA Simulation Environment & Offline Trajectory Data Collection (v2)
  -------------------------------------------------------------------------
  Research Context : Controllability Analysis of Decision Transformers in
                     Stochastic Wi-Fi / OFDMA Environments
  Author           : 松尾 春輝

  v2 changes (vs. wifi_env_data_collect.py)
  -----------------------------------------
  [ENV-1] reward_mode="queue_limited" (default): reward = packets actually
          served = min(Q_i, capacity_i).  Actions now change future queues and
          therefore future rewards (the legacy reward ignored the queue, which
          made the task an action-independent-future "bandit").
          reward_mode="legacy" reproduces the old behaviour exactly.
  [ENV-2] service_scale converts bit/Hz-like capacity into packets, so queues
          are no longer pinned at QUEUE_MAX.
  [ENV-3] Time-limit end is now `truncated=True` (terminated stays False).
  [POL-1] New expert policy "greedy" (exact one-step optimum for the reward).
          PF policy is rescaled so it is no longer an (almost) one-hot argmax.
  [DATA-1] Per-episode env seed = f(seed, episode_idx); policy choice, action
           noise and env noise use separate RNG streams.  Changing the expert
           ratio no longer changes the environment trajectories.
  [DATA-2] Expert/random split is exact (fixed counts) and nested across
           ratios (the 60 % expert set is a subset of the 80 % expert set).
  [DATA-3] collect_offline_dataset() accepts collision_rate, expert_policy,
           reward_mode, ...; ep_seeds are stored in the dataset.
  [RTG-1] Vectorised Local RTG with explicit tail handling
          ("truncate" = original behaviour, "rescale" = constant scale).
  [CHK-1] Stronger integrity checks (empirical collision rate, queue
          saturation, Local-RTG(k=T) == RTG, exact expert count).

  Dataset Schema (saved as .pkl)
  --------------------------------
  states[N,T,6] actions[N,T,3] rewards[N,T] dones[N,T] returns_to_go[N,T]
  policy_labels[N] ("pf"|"greedy"|"random")  ep_seeds[N]
  collision / arrivals / csi_base / queues / throughput / csi_noisy
  metadata (dict)
================================================================================
"""

import os
import time
import pickle
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

warnings.filterwarnings("ignore", category=DeprecationWarning, module="gymnasium")

# ------------------------------------------------------------------------------
# Global constants
# ------------------------------------------------------------------------------
NUM_USERS           = 3
CSI_MIN             = 1.0
CSI_MAX             = 10.0
QUEUE_MIN           = 0.0
QUEUE_MAX           = 50.0
DEFAULT_NOISE_LEVEL = 0.2        # sigma of instantaneous CSI noise
DEFAULT_COLLISION   = 0.2        # per-user collision probability

CSI_WALK_STD        = 0.15
CSI_WALK_MEAN_PULL  = 0.05

ARRIVAL_LAMBDA      = 3.0        # Poisson mean packets / user / step
# capacity (bit/Hz-like) -> packets.  2.0 keeps total demand (~9/step) and
# total capacity (~7-13/step) in the same range, so queues are not saturated.
SERVICE_SCALE       = 2.0

NUM_EPISODES        = 1000
EPISODE_LENGTH      = 100
DATASET_FILENAME    = "wifi_ofdma_offline_dataset.pkl"

PF_SOFTMAX_TEMP     = 0.25       # applied to mean-normalised PF scores
VALID_LABELS        = {"pf", "greedy", "random"}


# ==============================================================================
# 2.  WifiEnv
# ==============================================================================

class WifiEnv(gym.Env):
    """
    Wi-Fi / OFDMA resource-allocation environment.

    Observation : [CSI_base_1..3, Q_1..3]   (float32, shape (6,))
    Action      : raw box [0,1]^3, projected onto the simplex inside step().

    reward_mode
    -----------
    "queue_limited" : reward = sum_i min(Q_i, capacity_i) - drop_penalty * dropped
                      capacity_i = service_scale * a_i * log2(1 + 10*CSI_noisy_i),
                      zeroed on collision.  Serving an empty queue wastes
                      capacity, so actions have delayed consequences.
    "legacy"        : reward = sum_i capacity_i (queue ignored; original v1).

    Exogenous randomness (CSI noise, collisions, arrivals, CSI walk) is drawn
    with a fixed number of draws per step, independent of the action, so the
    same reset seed gives identical exogenous noise for every policy.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        noise_level   : float = DEFAULT_NOISE_LEVEL,
        collision_rate: float = DEFAULT_COLLISION,
        max_steps     : int   = EPISODE_LENGTH,
        seed          : Optional[int] = None,
        reward_mode   : str   = "queue_limited",
        service_scale : float = SERVICE_SCALE,
        arrival_lambda: float = ARRIVAL_LAMBDA,
        drop_penalty  : float = 0.0,
    ) -> None:
        super().__init__()
        assert reward_mode in ("queue_limited", "legacy"), reward_mode
        assert 0.0 <= collision_rate <= 1.0, "collision_rate must be in [0,1]"

        self.collision_rate = float(collision_rate)
        self.noise_level    = float(noise_level)
        self.max_steps      = int(max_steps)
        self.n_users        = NUM_USERS
        self.reward_mode    = reward_mode
        self.service_scale  = float(service_scale)
        self.arrival_lambda = float(arrival_lambda)
        self.drop_penalty   = float(drop_penalty)

        obs_low  = np.array([CSI_MIN]*self.n_users + [QUEUE_MIN]*self.n_users, dtype=np.float32)
        obs_high = np.array([CSI_MAX]*self.n_users + [QUEUE_MAX]*self.n_users, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.zeros(self.n_users, dtype=np.float32),
            high=np.ones(self.n_users, dtype=np.float32),
            dtype=np.float32,
        )

        self._csi_base   = np.zeros(self.n_users, dtype=np.float64)
        self._queue      = np.zeros(self.n_users, dtype=np.float64)
        self._step_count = 0
        self._needs_reset = True
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        super().reset(seed=seed)

        self._csi_base = self._rng.uniform(CSI_MIN, CSI_MAX, size=self.n_users)
        self._queue    = self._rng.uniform(QUEUE_MIN, QUEUE_MAX, size=self.n_users)
        self._step_count  = 0
        self._needs_reset = False
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert not self._needs_reset, "Call reset() before step()."

        # 1. Simplex projection
        action = np.clip(np.asarray(action, dtype=np.float64), 0.0, None)
        total  = action.sum()
        action = (np.ones(self.n_users) / self.n_users) if total < 1e-9 else action / total

        # 2. Exogenous randomness: fixed number of draws per step
        noise_csi  = self._rng.normal(0.0, self.noise_level, size=self.n_users)
        coll_draw  = self._rng.random(self.n_users)
        arrivals   = self._rng.poisson(self.arrival_lambda, size=self.n_users)
        walk_noise = self._rng.normal(0.0, CSI_WALK_STD, size=self.n_users)

        csi_noisy      = np.clip(self._csi_base + noise_csi, CSI_MIN, CSI_MAX)
        collision_mask = coll_draw < self.collision_rate

        # 3. Capacity / throughput
        rate = np.log2(1.0 + csi_noisy * 10.0)
        if self.reward_mode == "legacy":
            throughput = action * rate
            throughput[collision_mask] = 0.0
            served = throughput                       # legacy: queue not limiting
            reward = float(throughput.sum())
        else:
            capacity = self.service_scale * action * rate
            capacity[collision_mask] = 0.0
            served = np.minimum(self._queue, capacity)
            throughput = served

        # 4. Queue update (drop = overflow beyond QUEUE_MAX)
        q_after   = np.clip(self._queue - served, 0.0, None) + arrivals.astype(np.float64)
        dropped   = np.clip(q_after - QUEUE_MAX, 0.0, None)
        self._queue = np.clip(q_after, QUEUE_MIN, QUEUE_MAX)

        if self.reward_mode != "legacy":
            reward = float(served.sum() - self.drop_penalty * dropped.sum())

        # 5. Mean-reverting CSI walk (independent of the action)
        csi_mid = (CSI_MIN + CSI_MAX) / 2.0
        self._csi_base = np.clip(
            self._csi_base + CSI_WALK_MEAN_PULL * (csi_mid - self._csi_base) + walk_noise,
            CSI_MIN, CSI_MAX,
        )

        # 6. Time limit => truncated (not terminated)
        self._step_count += 1
        terminated = False
        truncated  = self._step_count >= self.max_steps
        if truncated:
            self._needs_reset = True

        info = {
            "throughput": throughput.copy(),
            "csi_noisy" : csi_noisy.copy(),
            "arrivals"  : arrivals.copy(),
            "dropped"   : dropped.copy(),
            "step"      : self._step_count,
            "collision" : bool(np.any(collision_mask)),
            "collision_mask": collision_mask.copy(),
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> None: pass
    def close(self) -> None: pass

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self._csi_base.astype(np.float32),
                               self._queue.astype(np.float32)])


# ==============================================================================
# 3.  Policies
# ==============================================================================

def _capacity_per_unit(csi: np.ndarray, service_scale: float) -> np.ndarray:
    return service_scale * np.log2(1.0 + csi * 10.0)


def policy_greedy(state: np.ndarray, rng: np.random.Generator,
                  service_scale: float = SERVICE_SCALE) -> np.ndarray:
    """
    Exact one-step optimum for the queue_limited reward (collision-agnostic):
    fill users in descending order of rate, each capped by the allocation that
    drains its queue (Q_i / rate_i).  Leftover allocation is spread uniformly.
    For reward_mode="legacy" this reduces to "all-in on the best CSI".
    """
    csi   = state[:NUM_USERS].astype(np.float64)
    queue = state[NUM_USERS:].astype(np.float64)
    c     = _capacity_per_unit(csi, service_scale)
    action, remaining = np.zeros(NUM_USERS), 1.0
    for i in np.argsort(-c):
        a_i = min(remaining, queue[i] / c[i])
        action[i] += a_i
        remaining -= a_i
        if remaining <= 1e-12:
            break
    if remaining > 1e-12:
        action += remaining / NUM_USERS
    return (action / action.sum()).astype(np.float32)


def policy_proportional_fair(state: np.ndarray, rng: np.random.Generator,
                             temperature: float = PF_SOFTMAX_TEMP) -> np.ndarray:
    """
    PF-style policy: score_i = CSI_i * (Q_i + 1), mean-normalised, then softmax.
    (v1 applied softmax to raw scores of magnitude up to ~500, which made the
    allocation an almost one-hot argmax.)
    """
    csi   = state[:NUM_USERS].astype(np.float64)
    queue = state[NUM_USERS:].astype(np.float64)
    score = csi * (queue + 1.0)
    score = score / score.mean()
    z = (score - score.max()) / temperature
    e = np.exp(z)
    return (e / e.sum()).astype(np.float32)


def policy_random_dirichlet(state: np.ndarray, rng: np.random.Generator,
                            alpha: float = 1.0) -> np.ndarray:
    return rng.dirichlet(np.full(NUM_USERS, alpha)).astype(np.float32)


EXPERT_POLICIES = {"greedy": policy_greedy, "pf": policy_proportional_fair}


# ==============================================================================
# 3.5  RTG / Local RTG
# ==============================================================================

def compute_returns_to_go(rewards: np.ndarray) -> np.ndarray:
    """Undiscounted RTG_t = sum_{t'>=t} r_t' for one episode (shape (T,))."""
    r = np.asarray(rewards, dtype=np.float64)
    return np.cumsum(r[::-1])[::-1].astype(np.float32)


def compute_local_rtg_for_dataset(rewards_array: np.ndarray, k: Optional[int],
                                  tail: str = "truncate") -> np.ndarray:
    """
    Local RTG  LRTG_t^(k) = sum_{i=t}^{min(t+k, T)-1} r_i   for [N, T] rewards.

    k=None (or k>=T) gives the ordinary RTG.

    tail
    ----
    "truncate": the last k-1 steps sum fewer than k rewards (v1 behaviour).
                The shrinking window is a deterministic artefact that mixes
                "horizon k" with "remaining steps".
    "rescale" : multiply the partial sum by k / (window length) so the scale
                matches a full window (an estimate of the k-step return).
    """
    assert tail in ("truncate", "rescale")
    r = np.asarray(rewards_array, dtype=np.float64)
    N, T = r.shape
    if k is None or k >= T:
        k = T
    c = np.concatenate([np.zeros((N, 1)), np.cumsum(r, axis=1)], axis=1)   # [N, T+1]
    t = np.arange(T)
    end = np.minimum(t + k, T)
    out = c[:, end] - c[:, t]
    if tail == "rescale":
        out = out * (k / (end - t))[None, :]
    return out.astype(np.float32)


# ==============================================================================
# 4.  Data collection
# ==============================================================================

def _episode_seed(seed: int, ep_idx: int) -> int:
    """Env seed depends only on (seed, episode index) -> identical across
    expert ratios / policies."""
    return int(np.random.SeedSequence([seed, 0, ep_idx]).generate_state(1)[0])


def assign_expert_episodes(num_episodes: int, expert_ratio: float, seed: int) -> np.ndarray:
    """
    Exact, nested assignment: a fixed random permutation (depending only on
    `seed`) ranks the episodes; the first round(N*ratio) are expert episodes.
    ratio 0.6 => subset of ratio 0.8.
    """
    perm = np.random.default_rng(np.random.SeedSequence([seed, 1])).permutation(num_episodes)
    n_expert = int(round(num_episodes * expert_ratio))
    is_expert = np.zeros(num_episodes, dtype=bool)
    is_expert[perm[:n_expert]] = True
    return is_expert


def collect_offline_dataset(
    num_episodes   : int   = NUM_EPISODES,
    episode_length : int   = EPISODE_LENGTH,
    noise_level    : float = DEFAULT_NOISE_LEVEL,
    collision_rate : float = DEFAULT_COLLISION,
    seed           : int   = 42,
    pf_split       : float = 0.8,           # expert ratio (kept name for compatibility)
    expert_policy  : str   = "greedy",      # "greedy" | "pf"
    reward_mode    : str   = "queue_limited",
    service_scale  : float = SERVICE_SCALE,
    drop_penalty   : float = 0.0,
    verbose        : bool  = True,
) -> Dict[str, Any]:
    """Collect an offline dataset (see module docstring for the schema)."""
    assert expert_policy in EXPERT_POLICIES, expert_policy
    expert_fn = EXPERT_POLICIES[expert_policy]
    if expert_policy == "greedy":
        expert_fn = lambda s, r: policy_greedy(s, r, service_scale=service_scale)

    N, T = num_episodes, episode_length
    all_states  = np.zeros((N, T, 6), dtype=np.float32)
    all_actions = np.zeros((N, T, 3), dtype=np.float32)
    all_rewards = np.zeros((N, T),    dtype=np.float32)
    all_dones   = np.zeros((N, T),    dtype=bool)
    all_rtg     = np.zeros((N, T),    dtype=np.float32)
    all_coll    = np.zeros((N, T),    dtype=bool)
    all_arr     = np.zeros((N, T, NUM_USERS), dtype=np.float32)
    all_thr     = np.zeros((N, T, NUM_USERS), dtype=np.float32)
    all_csin    = np.zeros((N, T, NUM_USERS), dtype=np.float32)
    all_csib    = np.zeros((N, T, NUM_USERS), dtype=np.float32)
    all_q       = np.zeros((N, T, NUM_USERS), dtype=np.float32)
    ep_seeds    = np.zeros(N, dtype=np.int64)

    is_expert = assign_expert_episodes(N, pf_split, seed)
    policy_labels = [expert_policy if e else "random" for e in is_expert]

    env = WifiEnv(noise_level=noise_level, collision_rate=collision_rate,
                  max_steps=T, reward_mode=reward_mode,
                  service_scale=service_scale, drop_penalty=drop_penalty)
    total_reward, start = 0.0, time.time()

    for ep in range(N):
        ep_seeds[ep] = _episode_seed(seed, ep)
        # action-noise RNG: separate stream per episode (never touches env RNG)
        act_rng = np.random.default_rng(np.random.SeedSequence([seed, 2, ep]))
        policy_fn = expert_fn if is_expert[ep] else policy_random_dirichlet

        obs, _ = env.reset(seed=int(ep_seeds[ep]))
        for t in range(T):
            all_states[ep, t] = obs
            all_csib[ep, t]   = env._csi_base
            all_q[ep, t]      = env._queue

            action = policy_fn(obs, act_rng)
            obs, reward, terminated, truncated, info = env.step(action)

            all_actions[ep, t] = action
            all_rewards[ep, t] = reward
            all_coll[ep, t]    = info["collision"]
            all_arr[ep, t]     = info["arrivals"]
            all_thr[ep, t]     = info["throughput"]
            all_csin[ep, t]    = info["csi_noisy"]
            all_dones[ep, t]   = terminated or truncated
            if terminated or truncated:
                break

        all_rtg[ep] = compute_returns_to_go(all_rewards[ep])
        total_reward += float(all_rewards[ep].sum())

        if verbose and (ep + 1) % 100 == 0:
            print(f"  [Episode {ep+1:>4d}/{N}] Avg-Reward: {total_reward/(ep+1):8.3f} "
                  f"| Elapsed: {time.time()-start:6.1f}s")
    env.close()

    n_exp = int(is_expert.sum())
    dataset = {
        "states": all_states, "actions": all_actions, "rewards": all_rewards,
        "dones": all_dones, "returns_to_go": all_rtg,
        "policy_labels": policy_labels, "ep_seeds": ep_seeds,
        "collision": all_coll, "arrivals": all_arr, "csi_base": all_csib,
        "queues": all_q, "throughput": all_thr, "csi_noisy": all_csin,
        "metadata": {
            "num_episodes": N, "episode_length": T,
            "noise_level": noise_level, "collision_rate": collision_rate,
            "num_users": NUM_USERS, "obs_dim": 6, "action_dim": 3,
            "expert_policy": expert_policy, "expert_ratio": pf_split,
            "pf_episodes": n_exp if expert_policy == "pf" else 0,
            "expert_episodes": n_exp, "random_episodes": N - n_exp,
            "reward_mode": reward_mode, "service_scale": service_scale,
            "drop_penalty": drop_penalty,
            "seed": seed, "arrival_lambda": ARRIVAL_LAMBDA,
            "csi_walk_std": CSI_WALK_STD, "pf_temperature": PF_SOFTMAX_TEMP,
            "total_reward": total_reward, "mean_ep_reward": total_reward / N,
        },
    }
    if verbose:
        print(f"\n  Collected {N} episodes: expert({expert_policy})={n_exp}, "
              f"random={N-n_exp}, mean ep-reward={total_reward/N:.3f}\n")
    return dataset


# ==============================================================================
# 5.  Serialisation
# ==============================================================================

def save_dataset(dataset: Dict[str, Any], filepath: str = DATASET_FILENAME) -> None:
    abs_path = os.path.abspath(filepath)
    with open(abs_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Dataset saved  : {abs_path}  ({os.path.getsize(abs_path)/1024**2:.3f} MB)")


def load_dataset(filepath: str = DATASET_FILENAME) -> Dict[str, Any]:
    abs_path = os.path.abspath(filepath)
    with open(abs_path, "rb") as f:
        dataset = pickle.load(f)
    print(f"  Dataset loaded : {abs_path}")
    return dataset


# ==============================================================================
# 6.  Integrity checks / summary
# ==============================================================================

def verify_dataset_integrity(dataset: Dict[str, Any]) -> None:
    print("  Running dataset integrity checks ...")
    meta = dataset["metadata"]
    N, T = meta["num_episodes"], meta["episode_length"]

    # 1. shapes
    assert dataset["states"].shape        == (N, T, 6)
    assert dataset["actions"].shape       == (N, T, 3)
    assert dataset["rewards"].shape       == (N, T)
    assert dataset["dones"].shape         == (N, T)
    assert dataset["returns_to_go"].shape == (N, T)
    assert len(dataset["policy_labels"])  == N

    # 2. simplex
    a = dataset["actions"].astype(np.float64)
    assert np.all(a >= -1e-6), "negative action component"
    assert np.allclose(a.sum(-1), 1.0, atol=1e-5), "actions do not sum to 1"

    # 3. bounds
    s = dataset["states"].astype(np.float64)
    assert np.all((s[..., :NUM_USERS] >= CSI_MIN - 1e-4) & (s[..., :NUM_USERS] <= CSI_MAX + 1e-4)), "CSI out of bounds"
    assert np.all((s[..., NUM_USERS:] >= QUEUE_MIN - 1e-4) & (s[..., NUM_USERS:] <= QUEUE_MAX + 1e-4)), "Queue out of bounds"

    # 4. RTG: independent recomputation + Local-RTG(k=T) == RTG
    r = dataset["rewards"].astype(np.float64)
    rtg_ref = np.cumsum(r[:, ::-1], axis=1)[:, ::-1]
    assert np.allclose(dataset["returns_to_go"], rtg_ref, rtol=1e-5, atol=1e-3), "RTG mismatch"
    assert np.allclose(compute_local_rtg_for_dataset(dataset["rewards"], None),
                       dataset["returns_to_go"], rtol=1e-5, atol=1e-3), "Local RTG(k=T) != RTG"
    assert np.allclose(compute_local_rtg_for_dataset(dataset["rewards"], 1), r, atol=1e-4), "Local RTG(k=1) != r"

    # 5. labels / exact expert count / unique env seeds
    labels = dataset["policy_labels"]
    assert all(l in VALID_LABELS for l in labels), "invalid policy label"
    if "expert_episodes" in meta:
        assert sum(l != "random" for l in labels) == meta["expert_episodes"], "expert count mismatch"
    if "ep_seeds" in dataset:
        assert len(set(dataset["ep_seeds"].tolist())) == N, "duplicate episode seeds"

    # 6. empirical collision rate (per user-step, 5-sigma binomial band)
    if "collision_rate" in meta and "collision" in dataset:
        p = meta["collision_rate"]
        n = N * T
        any_p = 1.0 - (1.0 - p) ** NUM_USERS            # P(at least one user collides)
        obs = dataset["collision"].mean()
        sd = np.sqrt(any_p * (1 - any_p) / n)
        assert abs(obs - any_p) < 5 * sd + 1e-3, f"collision freq {obs:.4f} vs expected {any_p:.4f}"

    # 7. queue saturation (informational warning)
    sat = float((dataset["queues"] >= QUEUE_MAX - 1e-3).mean())
    if sat > 0.3:
        print(f"  [WARN] {100*sat:.1f}% of (step,user) queues are at QUEUE_MAX -> "
              f"queue carries little information.")
    print("  All integrity checks passed.\n")


def print_dataset_summary(dataset: Dict[str, Any]) -> None:
    meta, rewards, rtg = dataset["metadata"], dataset["rewards"], dataset["returns_to_go"]
    labels = np.array(dataset["policy_labels"])
    ep_ret = rewards.sum(axis=1)
    print("=" * 70)
    print("  OFFLINE DATASET SUMMARY")
    print("=" * 70)
    print(f"  Episodes {meta['num_episodes']} | T {meta['episode_length']} | "
          f"reward_mode {meta.get('reward_mode','legacy')} | "
          f"collision {meta.get('collision_rate','?')} | noise {meta['noise_level']}")
    for lab in sorted(set(labels)):
        m = labels == lab
        print(f"  {lab:<7s}: {m.sum():>5d} eps ({100*m.mean():.1f}%)  "
              f"ep-return mean {ep_ret[m].mean():8.2f}  std {ep_ret[m].std():6.2f}")
    print(f"  Step reward : mean {rewards.mean():.4f} std {rewards.std():.4f}")
    print(f"  RTG[t=0]    : mean {rtg[:,0].mean():.4f} std {rtg[:,0].std():.4f}")
    print(f"  Queue at cap: {100*(dataset['queues']>=QUEUE_MAX-1e-3).mean():.1f}%")
    print("=" * 70 + "\n")


# ==============================================================================
# 7.  Smoke test
# ==============================================================================

def run_env_smoke_test() -> None:
    print("-" * 70)
    print("  Running WifiEnv smoke test ...")

    for mode in ("queue_limited", "legacy"):
        env = WifiEnv(max_steps=10, seed=0, reward_mode=mode)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (6,) and env.observation_space.contains(obs)

        rng = np.random.default_rng(1)
        res = env.step(rng.dirichlet([1, 1, 1]).astype(np.float32))
        assert len(res) == 5 and isinstance(res[1], float)
        assert "throughput" in res[4] and "collision_mask" in res[4]
        env.step(np.array([3.0, 7.0, 2.0], dtype=np.float32))   # unnormalised action

        env = WifiEnv(max_steps=100, seed=99, reward_mode=mode)
        env.reset()
        for i in range(100):
            _, _, terminated, truncated, _ = env.step(rng.dirichlet([1, 1, 1]))
            assert not terminated
            if truncated:
                assert i == 99, f"truncated early at step {i}"
                break
        assert truncated, "episode should be truncated at the time limit"

    # Exogenous noise identical for different policies under the same seed
    def exo(policy_fn):
        env = WifiEnv(max_steps=20)
        obs, _ = env.reset(seed=7)
        prng = np.random.default_rng(0)
        out = []
        for _ in range(20):
            obs, _, _, _, info = env.step(policy_fn(obs, prng))
            out.append(np.concatenate([info["csi_noisy"], info["arrivals"],
                                       info["collision_mask"].astype(float)]))
        return np.array(out)
    # csi_noisy depends on csi_base (action-independent), arrivals/collisions likewise
    assert np.allclose(exo(policy_greedy), exo(policy_random_dirichlet)), "exogenous noise depends on policy"

    # Action now matters for the future in queue_limited mode (and not in legacy)
    def future_diff(mode):
        # later actions are fixed (queue-independent) so only a_0 differs
        worst = 0.0
        for sd in range(10):
            res = []
            for first in (np.array([1., 0, 0]), np.array([0., 0, 1.])):
                env = WifiEnv(max_steps=30, reward_mode=mode)
                env.reset(seed=sd)
                rs = []
                for t in range(30):
                    a = first if t == 0 else np.array([0.2, 0.3, 0.5])
                    rs.append(env.step(a)[1])
                res.append(np.array(rs))
            worst = max(worst, np.abs(res[0][1:] - res[1][1:]).max())
        return worst
    assert future_diff("queue_limited") > 1e-6, "queue_limited: a_0 should affect future rewards"
    assert future_diff("legacy") < 1e-9, "legacy: future must be action-independent"

    # Local RTG
    r = np.random.default_rng(0).random((4, 100))
    assert np.allclose(compute_local_rtg_for_dataset(r, None), np.cumsum(r[:, ::-1], 1)[:, ::-1], atol=1e-4)
    assert np.allclose(compute_local_rtg_for_dataset(r, 1), r, atol=1e-6)
    ref = np.array([[r[e, t:t+5].sum() for t in range(100)] for e in range(4)])
    assert np.allclose(compute_local_rtg_for_dataset(r, 5), ref, atol=1e-4)

    print("  WifiEnv smoke test PASSED.")
    print("-" * 70 + "\n")


# ==============================================================================
# 8.  Main
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  Wi-Fi/OFDMA Offline RL Dataset Collection (v2)")
    print("=" * 70 + "\n")

    run_env_smoke_test()

    EXPERT_RATIO = 0.8
    print(f"  Config: {NUM_EPISODES} eps x {EPISODE_LENGTH} steps | "
          f"collision={DEFAULT_COLLISION} | noise={DEFAULT_NOISE_LEVEL} | "
          f"expert={int(EXPERT_RATIO*100)}% greedy / {int((1-EXPERT_RATIO)*100)}% random\n")

    dataset = collect_offline_dataset(
        num_episodes=NUM_EPISODES, episode_length=EPISODE_LENGTH,
        noise_level=DEFAULT_NOISE_LEVEL, collision_rate=DEFAULT_COLLISION,
        seed=2024, pf_split=EXPERT_RATIO, expert_policy="greedy",
        reward_mode="queue_limited", verbose=True,
    )
    print_dataset_summary(dataset)
    verify_dataset_integrity(dataset)

    save_dataset(dataset, DATASET_FILENAME)
    reloaded = load_dataset(DATASET_FILENAME)
    verify_dataset_integrity(reloaded)
    print("  Pipeline complete.\n")
