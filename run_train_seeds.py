# run_train_seeds.py
"""
Train DT and DDT_k for several TRAIN seeds with a fixed naming scheme.

    python run_train_seeds.py --dataset wifi_ofdma_offline_dataset.pkl \
        --ks global 1 2 3 5 10 --seeds 0 1 2 3 4 --outdir ckpt

Checkpoints: <outdir>/dt_wifi_model_{DT|DDT_k<k>}_s<seed>.pt  (skipped if they exist).
Evaluate with:
    python evaluate_DDT.py --dataset <same dataset> \
        --ckpts "ckpt/dt_wifi_model_DT_s*.pt" "ckpt/dt_wifi_model_DDT_k*_s*.pt" \
        --collisions 0.0 --num-seeds 50 --out-prefix eval_c0
(evaluate_DDT.py reads the train seed from the checkpoint's config.)
"""
import argparse
import os
from train_DDT_fixed import parse_args, train_ddt

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="wifi_ofdma_offline_dataset.pkl")
ap.add_argument("--ks", nargs="+", default=["global", "1", "2", "3", "5", "10"])
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
ap.add_argument("--outdir", default="ckpt")
ap.add_argument("--epochs", type=int, default=30)
a = ap.parse_args()

os.makedirs(a.outdir, exist_ok=True)
for k in a.ks:
    tag = "DT" if k.lower() in ("global", "none", "t") else f"DDT_k{int(k)}"
    for s in a.seeds:
        out = os.path.join(a.outdir, f"dt_wifi_model_{tag}_s{s}.pt")
        if os.path.exists(out):
            print(f"skip {out}")
            continue
        train_ddt(parse_args(["--dataset", a.dataset, "--k", k, "--seed", str(s),
                              "--epochs", str(a.epochs), "--out", out]))
