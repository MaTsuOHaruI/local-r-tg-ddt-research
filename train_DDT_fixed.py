# train_DDT_fixed.py
"""
Train DT (global RTG) or DDT (Local RTG, horizon k) with ONE script.

    python train_DDT_fixed.py --k global            # DT      -> dt_wifi_model_DT.pt
    python train_DDT_fixed.py --k 5                 # DDT k=5 -> dt_wifi_model_DDT_k5.pt
    python train_DDT_fixed.py --k-min 1 --k-max 50 --use-k-cond   # variable k + adaLN

Fixes vs. the previous version
------------------------------
[T-1] Uses the fixed models (no action leak): loss is computed on logits of the
      softmax policy with SOFT CROSS-ENTROPY against the simplex action, the same
      softmax that evaluation uses (previously: MSE on raw outputs, softmax only
      at evaluation).
[T-2] The checkpoint stores {state_dict, config}; evaluation rebuilds the model
      (k, RTG normalisation, tail handling, ...) from the config -> no silent
      mismatch between training and evaluation.
[T-3] k_val is only fed to the network when it is informative (k_min < k_max).
      Default DDT = the plain DT architecture with a Local RTG input.
[T-4] Several windows per episode per epoch (see dataset_fixed.py), seeds fixed.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_fixed import WifiOfflineDataset
from model_DDT import build_model_from_config


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wifi_ofdma_offline_dataset.pkl")
    p.add_argument("--k", default="50", help="'global' (DT) or an integer horizon k")
    p.add_argument("--k-min", type=int, default=None)
    p.add_argument("--k-max", type=int, default=None)
    p.add_argument("--use-k-cond", action="store_true", help="adaLN k-conditioning (needs k-min<k-max)")
    p.add_argument("--tail", default="truncate", choices=["truncate", "rescale"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--windows", type=int, default=20, help="windows per episode per epoch")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--context", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=0.25)
    p.add_argument("--n-layer", type=int, default=3)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0, help="train seed")
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def train_ddt(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    is_global = str(args.k).lower() in ("global", "none", "t")
    if is_global:
        mode, k_min, k_max, arch = "global", 0, 0, "dt"
    else:
        k_min = args.k_min if args.k_min is not None else int(args.k)
        k_max = args.k_max if args.k_max is not None else int(args.k)
        mode, arch = "local_k", "ddt"
    use_k_cond = bool(args.use_k_cond) and (not is_global) and k_min < k_max
    if args.use_k_cond and not use_k_cond:
        print("[WARN] --use-k-cond ignored (needs Local RTG with k_min < k_max)")

    dataset = WifiOfflineDataset(
        args.dataset, context_len=args.context, mode=mode, k_min=k_min, k_max=k_max,
        tail=args.tail, windows_per_episode=args.windows, seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    cfg = dict(
        arch=arch, state_dim=dataset.states.shape[-1], action_dim=dataset.actions.shape[-1],
        max_ep_len=dataset.max_len, n_layer=args.n_layer, n_head=args.n_head,
        n_embd=args.n_embd, dropout=args.dropout, use_k_cond=use_k_cond,
        mode=mode, k=None if is_global else (k_min if k_min == k_max else None),
        k_min=None if is_global else k_min, k_max=None if is_global else k_max,
        tail=args.tail, context_len=args.context, mean_reward=dataset.mean_reward,
        seed=args.seed, dataset_path=args.dataset, loss="soft_ce",
    )
    model = build_model_from_config(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    tag = "DT" if is_global else (f"DDT_k{k_min}" if k_min == k_max else f"DDT_k{k_min}-{k_max}")
    out = args.out or f"dt_wifi_model_{tag}.pt"
    print(f"Device {device} | model {tag} | steps/epoch {len(loader)} | out {out}")

    for epoch in range(args.epochs):
        model.train()
        tot, tot_p, n = 0.0, 0.0, 0
        for states, actions, rtgs, timesteps, masks, k_used in loader:
            states, actions, rtgs = states.to(device), actions.to(device), rtgs.to(device)
            timesteps, masks, k_used = timesteps.to(device), masks.to(device), k_used.to(device)

            logits = model(states, actions, rtgs, timesteps,
                           k_val=k_used if use_k_cond else None)
            logp = F.log_softmax(logits, dim=-1)
            ce = -(actions * logp).sum(-1)                        # [B,K]
            loss = (ce * masks).sum() / masks.sum()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()

            tot += loss.item()
            tot_p += ((logp.exp().max(-1).values) * masks).sum().item() / masks.sum().item()
            n += 1
        print(f"Epoch [{epoch+1}/{args.epochs}] | soft-CE {tot/n:.4f} | mean max-prob {tot_p/n:.3f}")

    torch.save({"state_dict": model.state_dict(), "config": cfg}, out)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    train_ddt(parse_args())
