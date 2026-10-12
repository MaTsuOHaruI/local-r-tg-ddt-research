# baselines_eval.py
"""
Reference policies for the Wi-Fi env, written in the SAME raw-CSV format as
evaluate_DDTv2.py (columns: model, base, collision, mult, seed, train_seed, ret, max_action),
so the output can simply be concatenated with evaluation_DDT_raw.csv and fed to the
existing paired analysis.

    uniform      : a = (1/3, 1/3, 1/3)
    mix_l<lam>   : a = lam * greedy(s) + (1 - lam) * uniform
                   (lam = 1 is greedy, lam = 0 is uniform)

Why: the DT/DDT policy is softmax(logits) trained with soft cross-entropy, i.e. it outputs the
conditional MEAN of the data actions.  On an 80 % greedy / 20 % random dataset the mean action is
approximately lam*greedy + (1-lam)*uniform, so this family is the natural "no-RTG" yardstick.

    python baselines_eval.py --dataset wifi_ofdma_offline_dataset.pkl --out baselines_raw.csv
    # then
    import pandas as pd
    raw = pd.concat([pd.read_csv("evaluation_DDT_raw.csv"), pd.read_csv("baselines_raw.csv")])

Evaluation seeds / env construction mirror evaluate_DDTv2.py (seed-base 10000, 50 seeds,
noise 0.2, queue_limited, service_scale 2.0 unless the dataset metadata says otherwise).
"""
import argparse
import inspect
import pickle
import numpy as np
import pandas as pd

from wifi_env_data_collect_v2 import WifiEnv, policy_greedy


def make_env(meta, collision_rate):
    wanted = dict(
        noise_level=meta.get("noise_level", 0.2), collision_rate=collision_rate,
        max_steps=meta.get("episode_length", 100), reward_mode=meta.get("reward_mode", "queue_limited"),
        service_scale=meta.get("service_scale", 2.0), drop_penalty=meta.get("drop_penalty", 0.0),
    )
    sig = inspect.signature(WifiEnv.__init__).parameters
    return WifiEnv(**{k: v for k, v in wanted.items() if k in sig and v is not None})


def run_policy(meta, col, seeds, act_fn):
    T = meta.get("episode_length", 100)
    rets, mxs = [], []
    for s in seeds:
        env = make_env(meta, col)
        obs, _ = env.reset(seed=int(s))
        R = m = 0.0
        for _ in range(T):
            a = act_fn(obs)
            m += float(a.max())
            obs, r, _, _, _ = env.step(a)
            R += r
        rets.append(R)
        mxs.append(m / T)
    return np.array(rets), np.array(mxs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None, help="only its metadata is used (optional)")
    p.add_argument("--collisions", nargs="+", type=float, default=[0.0, 0.3, 0.6, 0.9])
    p.add_argument("--lams", nargs="+", type=float, default=[0.5, 0.8, 0.9])
    p.add_argument("--num-seeds", type=int, default=50)
    p.add_argument("--seed-base", type=int, default=10_000)
    p.add_argument("--out", default="baselines_raw.csv")
    args = p.parse_args()

    meta = {}
    if args.dataset:
        with open(args.dataset, "rb") as f:
            meta = pickle.load(f)["metadata"]
    scale = meta.get("service_scale", 2.0)
    seeds = args.seed_base + np.arange(args.num_seeds)

    def mix(lam):
        return lambda o: (lam * policy_greedy(o, None, service_scale=scale) + (1.0 - lam) / 3.0).astype(np.float32)

    policies = [("uniform", mix(0.0))] + [(f"mix_l{lam:g}", mix(lam)) for lam in args.lams]
    rows = []
    for col in args.collisions:
        for name, fn in policies:
            ret, mx = run_policy(meta, col, seeds, fn)
            rows += [dict(model=name, base=name, collision=col, mult=1.0, seed=int(s), train_seed=-1,
                          ret=float(r), max_action=float(m)) for s, r, m in zip(seeds, ret, mx)]
            print(f"collision {col:.1f} | {name:<10s} return {ret.mean():8.2f} ± {ret.std(ddof=1)/np.sqrt(len(ret)):.2f}"
                  f" | mean max-action {mx.mean():.2f}")
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
