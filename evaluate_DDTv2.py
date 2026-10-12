# evaluate_DDTv2.py  (v3.1: multi-train-seed aware)
"""
Paired evaluation of DT / DDT(k) checkpoints (+ greedy / random reference policies).

    python evaluate_DDT.py --ckpts dt_wifi_model_DT.pt "dt_wifi_model_DDT_k*.pt" \
        --collisions 0.0 0.3 0.6 0.9 --num-seeds 50

New in v3
---------
* DT is evaluated with BOTH target-RTG rules, as two separate rows:
    "DT"      : rtg_mode=profile    (target = dataset quantile profile; realised rewards
                                     never enter the input; same rule as DDT)
    "DT_dec"  : rtg_mode=decrement  (textbook DT: R_t = R_0 - sum of realised rewards,
                                     so collision noise DOES enter the input)
  DDT always uses "profile": a sliding k-step Local RTG has no canonical decrement rule.
  The contrast DT_dec vs DT / DDT_k is the one that tests the "noisy long RTG" hypothesis.
  (--no-dt-decrement switches DT_dec off.)
* --rtg-mults 0.5 1.0 1.5 : multiplies the target RTG; return vs. multiplier is plotted
  (closed-loop RTG controllability).  Rows with mult != 1 are labelled "<model>@x<mult>".
* Models are ordered DT, DT_dec, DDT_k1, k5, k10, ... (numeric), then greedy, random.
* Figure: (1) return (95% CI over eval seeds), (2) normalised score
  (return - random)/(greedy - random), (3) paired difference vs --ref (95% CI).
  All error bars are 95% CI = 1.96 x SEM over evaluation seeds.
  NOTE: the CI does not include training-seed variance (train several seeds for that).

New in v3.1
-----------
* Several checkpoints with the same label (same model, different TRAIN seed, e.g.
  dt_wifi_model_DDT_k3_s0.pt, ..._s1.pt) are evaluated separately; the train seed is
  read from cfg["seed"] and stored in the raw CSV column `train_seed`
  (greedy / random reference rows have train_seed = -1).
* Summary / plots first average over train seeds per evaluation seed (no
  pseudo-replication: n = number of eval seeds, not eval seeds x train seeds).
  A per-train-seed table is written to <prefix>_by_train_seed.csv.
* Warns when a checkpoint was trained on a different dataset file than --dataset
  (the target-RTG profile is computed from --dataset).
"""
import argparse
import glob
import inspect
import os
import pickle
import re
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_fixed import local_rtg_array
from model_DDT import build_model_from_config

try:
    from wifi_env_data_collect_v2 import WifiEnv, policy_greedy
except ImportError:                                   # v1 env: no greedy baseline
    from wifi_env_data_collect import WifiEnv
    policy_greedy = None


# ------------------------------------------------------------------ helpers
def make_env(meta, collision_rate):
    wanted = dict(
        noise_level=meta.get("noise_level", 0.2), collision_rate=collision_rate,
        max_steps=meta["episode_length"], reward_mode=meta.get("reward_mode", "legacy"),
        service_scale=meta.get("service_scale"), drop_penalty=meta.get("drop_penalty"),
    )
    sig = inspect.signature(WifiEnv.__init__).parameters
    return WifiEnv(**{k: v for k, v in wanted.items() if k in sig and v is not None})


def target_profile(rewards, k, tail, q):
    return np.quantile(local_rtg_array(rewards, k, tail), q, axis=0)          # [T]


def softmax_np(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def sort_key(label):
    base = label.split("@")[0]
    if base == "DT":
        return (0, 0, label)
    if base == "DT_dec":
        return (1, 0, label)
    m = re.match(r"DDT(?:var)?_k(\d+)", base)
    if m:
        return (2, int(m.group(1)), label)
    return (3, {"greedy": 0, "random": 1}.get(base, 2), label)


class ModelPolicy:
    def __init__(self, model, cfg, k_eval, profile, rtg_mode, mult, device):
        self.m, self.cfg, self.dev = model, cfg, device
        self.k_eval = cfg["max_ep_len"] if k_eval is None else int(k_eval)   # global -> T
        self.profile, self.rtg_mode, self.mult = profile, rtg_mode, mult
        self.norm = self.k_eval * cfg["mean_reward"]
        self.ctx = cfg["context_len"]

    def reset(self, B, T):
        self.R = np.zeros((B, T, 1), np.float32)
        self.dec = np.full(B, self.profile[0] * self.mult)

    def __call__(self, t, obs, S, A):
        B = len(obs)
        tgt = np.full(B, self.profile[t] * self.mult) if self.rtg_mode == "profile" else self.dec.copy()
        self.R[:, t, 0] = tgt / self.norm
        t0 = max(0, t - self.ctx + 1)
        ts = torch.arange(t0, t + 1, device=self.dev).unsqueeze(0).repeat(B, 1)
        kv = torch.full((B,), float(self.k_eval), device=self.dev) if self.cfg["use_k_cond"] else None
        with torch.no_grad():
            logits = self.m(torch.from_numpy(S[:, t0:t + 1]).to(self.dev),
                            torch.from_numpy(A[:, t0:t + 1]).to(self.dev),
                            torch.from_numpy(self.R[:, t0:t + 1]).to(self.dev), ts, k_val=kv)
        return softmax_np(logits[:, -1].cpu().numpy().astype(np.float64)).astype(np.float32)

    def observe_reward(self, r):                          # decrement mode: R_{t+1} = R_t - r_t
        self.dec -= r


class SimplePolicy:
    def __init__(self, fn, seed_base):
        self.fn, self.seed_base = fn, seed_base

    def reset(self, B, T):
        self.rngs = [np.random.default_rng(self.seed_base + i) for i in range(B)]

    def __call__(self, t, obs, S, A):
        return np.stack([self.fn(o, r) for o, r in zip(obs, self.rngs)])


def run_batch(env_fn, seeds, T, pol):
    """All seeds in one batch.  Returns episode returns [B] and mean max action [B]."""
    envs = [env_fn() for _ in seeds]
    obs = np.stack([e.reset(seed=int(s))[0] for e, s in zip(envs, seeds)])
    B = len(seeds)
    S = np.zeros((B, T, obs.shape[1]), np.float32)
    A = np.zeros((B, T, 3), np.float32)           # A[:, t] is a zero placeholder when the model reads S_t
    ret, mx = np.zeros(B), np.zeros(B)
    pol.reset(B, T)
    for t in range(T):
        S[:, t] = obs
        a = pol(t, obs, S, A)
        A[:, t] = a
        mx += a.max(-1)
        rs = np.zeros(B)
        for i, e in enumerate(envs):
            o, r, term, trunc, _ = e.step(a[i])
            obs[i], rs[i] = o, r
        ret += rs
        if hasattr(pol, "observe_reward"):
            pol.observe_reward(rs)
    return ret, mx / T


# ------------------------------------------------------------------ main
def load_entries(args, device):
    ckpts = []
    for pat in args.ckpts:
        ckpts += sorted(glob.glob(pat)) or [pat]
    entries = []                       # (label, model, cfg, k_eval, rtg_mode)
    for path in ckpts:
        ck = torch.load(path, map_location=device)
        cfg = ck["config"]
        model = build_model_from_config(cfg).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        if cfg["arch"] == "dt":
            entries.append(("DT", model, cfg, None, "profile"))
            if args.dt_decrement:
                entries.append(("DT_dec", model, cfg, None, "decrement"))
        elif cfg["k_min"] == cfg["k_max"]:
            entries.append((f"DDT_k{cfg['k_min']}", model, cfg, cfg["k_min"], "profile"))
        else:
            for k in args.eval_ks:
                entries.append((f"DDTvar_k{k}", model, cfg, k, "profile"))
    return entries


def evaluate(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with open(args.dataset, "rb") as f:
        data = pickle.load(f)
    meta, rewards = data["metadata"], data["rewards"]
    T = meta["episode_length"]
    seeds = args.seed_base + np.arange(args.num_seeds)
    entries = load_entries(args, device)
    for label, _m, cfg, *_ in entries:
        dp = cfg.get("dataset_path")
        if dp and os.path.basename(dp) != os.path.basename(args.dataset):
            print(f"[WARN] {label}: trained on '{dp}' but target RTG profile / env meta come from "
                  f"'{args.dataset}'")

    rows = []
    def add(label, col, mult, ret, mx, tseed=-1):
        tag = label if mult == 1.0 else f"{label}@x{mult:g}"
        rows.extend(dict(model=tag, base=label, collision=col, mult=mult, seed=int(s),
                         train_seed=int(tseed), ret=r, max_action=m)
                    for s, r, m in zip(seeds, ret, mx))
        ts = f" [train seed {tseed}]" if tseed >= 0 else ""
        print(f"  {tag + ts:<26s} return {ret.mean():8.2f} ± {ret.std(ddof=1)/np.sqrt(len(ret)):.2f} (SEM) "
              f"| mean max-action {mx.mean():.2f}")

    for col in args.collisions:
        env_fn = lambda: make_env(meta, col)
        print(f"\n[collision {col:.2f}]")
        for label, model, cfg, k_eval, mode in entries:
            k_prof = None if cfg["arch"] == "dt" else k_eval
            prof = target_profile(rewards, k_prof, cfg["tail"], args.quantile)
            for mult in args.rtg_mults:
                pol = ModelPolicy(model, cfg, k_eval, prof, mode, mult, device)
                add(label, col, mult, *run_batch(env_fn, seeds, T, pol), tseed=cfg.get("seed", 0))
        refs = [("random", SimplePolicy(lambda o, r: r.dirichlet(np.ones(3)).astype(np.float32), 777))]
        if policy_greedy is not None:
            kw = {"service_scale": meta["service_scale"]} if "service_scale" in meta else {}
            refs.insert(0, ("greedy", SimplePolicy(lambda o, r: policy_greedy(o, r, **kw), 0)))
        for label, pol in refs:
            add(label, col, 1.0, *run_batch(env_fn, seeds, T, pol))

    raw = pd.DataFrame(rows)
    raw.to_csv(f"{args.out_prefix}_raw.csv", index=False)
    summary_and_plots(raw, args)
    return raw


def summary_and_plots(raw, args):
    cols = list(args.collisions)
    if "train_seed" in raw:
        (raw.groupby(["collision", "model", "base", "mult", "train_seed"], as_index=False)
            .agg(ret=("ret", "mean"), std_over_eval_seeds=("ret", "std"), n_eval=("ret", "size"))
            .to_csv(f"{args.out_prefix}_by_train_seed.csv", index=False))
        # average over train seeds per evaluation seed -> one value per (model, eval seed)
        raw = (raw.groupby(["collision", "model", "base", "mult", "seed"], as_index=False)
                  .agg(ret=("ret", "mean"), max_action=("max_action", "mean"),
                       n_train=("train_seed", "nunique")))
    g = raw.groupby(["collision", "model"])
    summ = g.agg(mean=("ret", "mean"), std=("ret", "std"), n=("ret", "size"),
                 mean_max_action=("max_action", "mean")).reset_index()
    summ["sem"] = summ["std"] / np.sqrt(summ["n"])
    summ.to_csv(f"{args.out_prefix}_summary.csv", index=False)

    main = raw[raw.mult == 1.0]
    order = sorted(main["model"].unique(), key=sort_key)
    piv = main.pivot_table(index=["collision", "seed"], columns="model", values="ret")

    # normalised score: (ret - random) / mean(greedy - random), paired over seeds
    norm = {}
    if "greedy" in piv and "random" in piv:
        for c in cols:
            sub = piv.loc[c]
            denom = (sub["greedy"] - sub["random"]).mean()
            for m in order:
                v = (sub[m] - sub["random"]) / denom
                norm[(c, m)] = (v.mean(), v.std(ddof=1) / np.sqrt(len(v)), denom)

    ref = args.ref if args.ref in order else None
    paired = []
    if ref:
        for c in cols:
            for m in order:
                if m == ref:
                    continue
                d = (piv.loc[c][m] - piv.loc[c][ref]).dropna()
                paired.append(dict(collision=c, model=m, ref=ref, mean_diff=d.mean(),
                                   sem=d.std(ddof=1) / np.sqrt(len(d)), n=len(d)))
        pd.DataFrame(paired).to_csv(f"{args.out_prefix}_paired_vs_{ref}.csv", index=False)

    panels = 1 + bool(norm) + bool(ref)
    fig, ax = plt.subplots(1, panels, figsize=(6 * panels, 5), squeeze=False)
    cmap = plt.get_cmap("viridis")
    colour = lambda i: cmap(i / max(1, len(cols) - 1))
    xs = np.arange(len(order))
    p = 0
    for ci, c in enumerate(cols):
        sub = summ[summ.collision == c].set_index("model").reindex(order)
        ax[0, p].errorbar(xs, sub["mean"], yerr=1.96 * sub["sem"], marker="o", capsize=3,
                          color=colour(ci), label=f"collision {c:.1f}")
    ax[0, p].set_title("Mean return (95% CI)"); ax[0, p].set_ylabel("return"); p += 1
    if norm:
        for ci, c in enumerate(cols):
            y = [norm[(c, m)][0] for m in order]; e = [1.96 * norm[(c, m)][1] for m in order]
            ax[0, p].errorbar(xs, y, yerr=e, marker="o", capsize=3, color=colour(ci), label=f"collision {c:.1f}")
        ax[0, p].axhline(0, color="gray", lw=.8, ls=":"); ax[0, p].axhline(1, color="gray", lw=.8, ls=":")
        ax[0, p].set_title("Normalised: 0 = random, 1 = greedy (95% CI)"); p += 1
    if ref:
        pv = pd.DataFrame(paired)
        for ci, c in enumerate(cols):
            sub = pv[pv.collision == c].set_index("model").reindex(order)
            ax[0, p].errorbar(xs, sub["mean_diff"], yerr=1.96 * sub["sem"], marker="o", capsize=3,
                              color=colour(ci), label=f"collision {c:.1f}")
        ax[0, p].axhline(0, color="k", lw=.8)
        ax[0, p].set_title(f"Paired difference vs {ref} (95% CI)")
    for a in ax[0]:
        a.set_xticks(xs); a.set_xticklabels(order, rotation=45, ha="right")
        a.grid(alpha=.4); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{args.out_prefix}.png", dpi=200); plt.close()

    # RTG multiplier sweep (closed-loop controllability)
    if raw["mult"].nunique() > 1:
        mults = sorted(raw["mult"].unique())
        bases = sorted([b for b in raw["base"].unique() if b not in ("greedy", "random")], key=sort_key)
        fig, ax = plt.subplots(1, len(cols), figsize=(4.5 * len(cols), 4), squeeze=False)
        for ci, c in enumerate(cols):
            for b in bases:
                s = raw[(raw.collision == c) & (raw.base == b)].groupby("mult")["ret"].agg(["mean", "std", "size"])
                ax[0, ci].errorbar(s.index, s["mean"], yerr=1.96 * s["std"] / np.sqrt(s["size"]),
                                   marker="o", capsize=3, label=b)
            ax[0, ci].set_title(f"collision {c:.1f}"); ax[0, ci].set_xlabel("target RTG multiplier")
            ax[0, ci].grid(alpha=.4)
        ax[0, 0].set_ylabel("return"); ax[0, 0].legend(fontsize=7)
        plt.tight_layout(); plt.savefig(f"{args.out_prefix}_rtg_mult.png", dpi=200); plt.close()
    print(f"\nSaved {args.out_prefix}_raw.csv / _summary.csv / .png")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wifi_ofdma_offline_dataset.pkl")
    p.add_argument("--ckpts", nargs="+", default=["dt_wifi_model_DT.pt", "dt_wifi_model_DDT_k*.pt"])
    p.add_argument("--collisions", nargs="+", type=float, default=[0.0, 0.3, 0.6, 0.9])
    p.add_argument("--num-seeds", type=int, default=50)
    p.add_argument("--seed-base", type=int, default=10_000)
    p.add_argument("--quantile", type=float, default=0.9, help="target RTG quantile taken from the dataset")
    p.add_argument("--rtg-mults", nargs="+", type=float, default=[1.0], help="target RTG multipliers")
    p.add_argument("--no-dt-decrement", dest="dt_decrement", action="store_false",
                   help="skip the DT_dec row")
    p.add_argument("--eval-ks", nargs="+", type=int, default=[1, 5, 10, 20, 50],
                   help="k values for variable-k checkpoints")
    p.add_argument("--ref", default="DT", help="reference label for paired differences")
    p.add_argument("--out-prefix", default="evaluation_DDT")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())