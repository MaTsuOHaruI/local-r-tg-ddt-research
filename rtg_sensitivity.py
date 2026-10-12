# rtg_sensitivity.py
"""
Offline diagnostic: does the trained model actually USE the RTG input?

For windows sampled from the dataset, the model's action distribution is computed
with (a) the true RTG and (b) modified RTGs (x0, x0.5, x1.5, x2, and RTG taken from a
different window).  We report the mean total-variation distance
        TV = 0.5 * sum_a |p_true(a) - p_modified(a)|     (0 = identical, 1 = disjoint)
between the two action distributions.  No environment roll-out is needed.

    python rtg_sensitivity.py --ckpts dt_wifi_model_DT.pt dt_wifi_model_DDT_k*.pt

Reading the result
------------------
TV(x0.5), TV(x1.5) << 0.01 : the model ignores the RTG (RTG controllability ~ 0).
"fit TV" is TV(model, dataset action): the scale of the model's own prediction error;
RTG effects should be judged relative to it.
"""
import argparse
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset_fixed import WifiOfflineDataset
from model_DDT import build_model_from_config

VARIANTS = {"x0": 0.0, "x0.5": 0.5, "x1.5": 1.5, "x2": 2.0}


def tv(p, q):
    return 0.5 * (p - q).abs().sum(-1)


@torch.no_grad()
def analyse(path, dataset_path, n_windows, device, seed):
    ck = torch.load(path, map_location=device)
    cfg = ck["config"]
    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    glob_mode = cfg["mode"] == "global"
    ds = WifiOfflineDataset(
        dataset_path or cfg["dataset_path"], context_len=cfg["context_len"], mode=cfg["mode"],
        k_min=0 if glob_mode else cfg["k_min"], k_max=0 if glob_mode else cfg["k_max"],
        tail=cfg["tail"], windows_per_episode=max(1, n_windows // 1000 + 1), seed=seed,
    )
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    acc = {name: [] for name in list(VARIANTS) + ["shuffled", "fit"]}
    total = 0
    for states, actions, rtgs, timesteps, masks, k_used in loader:
        states, actions, rtgs = states.to(device), actions.to(device), rtgs.to(device)
        timesteps, masks, k_used = timesteps.to(device), masks.to(device), k_used.to(device)
        kv = k_used if cfg["use_k_cond"] else None
        p0 = torch.softmax(model(states, actions, rtgs, timesteps, k_val=kv), -1)
        w = masks
        acc["fit"].append(((tv(p0, actions) * w).sum() / w.sum()).item())
        for name, m in VARIANTS.items():
            p1 = torch.softmax(model(states, actions, rtgs * m, timesteps, k_val=kv), -1)
            acc[name].append(((tv(p0, p1) * w).sum() / w.sum()).item())
        p1 = torch.softmax(model(states, actions, torch.roll(rtgs, 1, dims=0), timesteps, k_val=kv), -1)
        acc["shuffled"].append(((tv(p0, p1) * w).sum() / w.sum()).item())
        total += len(states)
        if total >= n_windows:
            break
    row = {k: float(np.mean(v)) for k, v in acc.items()}
    row["model"] = "DT" if cfg["arch"] == "dt" else (
        f"DDT_k{cfg['k_min']}" if cfg["k_min"] == cfg["k_max"] else f"DDTvar_k{cfg['k_min']}-{cfg['k_max']}")
    return row


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", default=["dt_wifi_model_DT.pt", "dt_wifi_model_DDT_k*.pt"])
    p.add_argument("--dataset", default=None, help="default: path stored in the checkpoint")
    p.add_argument("--n-windows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="rtg_sensitivity.csv")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = []
    for pat in a.ckpts:
        paths += sorted(glob.glob(pat)) or [pat]
    rows = [analyse(p, a.dataset, a.n_windows, device, a.seed) for p in paths]
    df = pd.DataFrame(rows)[["model", "fit", "x0", "x0.5", "x1.5", "x2", "shuffled"]]
    df = df.rename(columns={"fit": "fit_TV"})
    pd.set_option("display.width", 140)
    print(df.round(4).to_string(index=False))
    df.to_csv(a.out, index=False)
    print(f"\nSaved {a.out}")
