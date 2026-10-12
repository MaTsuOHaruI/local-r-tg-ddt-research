# make_stats_csv.py
"""
evaluate_DDT.py raw CSV  ->  CSV for stats_paired.py (columns: collision, model, seed, return)

    python make_stats_csv.py evaluation_DDT_raw.csv --unit train --out raw_results.csv
    python stats_paired.py raw_results.csv --ref DT --dt-as-k 100 --out out_stats

--unit train (recommended for claims about the METHOD)
    one row per (collision, model, TRAIN seed): return = mean over evaluation seeds.
    `seed` = train seed, so paired tests pair DDT_k with DT trained with the same seed.
    n = number of train seeds (3-5): honest, but low power (see g_power.csv).
--unit eval
    one row per (collision, model, EVAL seed): return = mean over train seeds.
    n = number of eval seeds: more power, but the CI covers only evaluation noise for
    this particular set of trained models (it does NOT include train-seed variance).

DT_dec (decrement RTG rule) and RTG-multiplier rows (mult != 1) are dropped by default;
use --keep-dt-dec to keep DT_dec as its own model (e.g. for a DT_dec vs DT test).
greedy / random (train_seed = -1) are copied to every train seed in --unit train.
"""
import argparse
import pandas as pd


def convert(raw: pd.DataFrame, unit: str, keep_dt_dec: bool = False) -> pd.DataFrame:
    df = raw[raw["mult"] == 1.0].copy()
    if not keep_dt_dec:
        df = df[df["model"] != "DT_dec"]
    if "train_seed" not in df:
        df["train_seed"] = 0
    refs = df[df["train_seed"] < 0]
    trained = df[df["train_seed"] >= 0]

    if unit == "eval":
        out = (pd.concat([trained, refs])
                 .groupby(["collision", "model", "seed"], as_index=False)["ret"].mean())
    else:
        tr = (trained.groupby(["collision", "model", "train_seed"], as_index=False)["ret"].mean()
                     .rename(columns={"train_seed": "seed"}))
        rf = (refs.groupby(["collision", "model"], as_index=False)["ret"].mean())
        seeds = sorted(trained["train_seed"].unique())
        rf = pd.concat([rf.assign(seed=s) for s in seeds], ignore_index=True) if len(rf) else rf
        out = pd.concat([tr, rf], ignore_index=True)
    out = out.rename(columns={"ret": "return"})
    return out[["collision", "model", "seed", "return"]].sort_values(
        ["collision", "model", "seed"]).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_csv")
    ap.add_argument("--unit", choices=["train", "eval"], default="train")
    ap.add_argument("--keep-dt-dec", action="store_true")
    ap.add_argument("--out", default="raw_results.csv")
    a = ap.parse_args()
    out = convert(pd.read_csv(a.raw_csv), a.unit, a.keep_dt_dec)
    out.to_csv(a.out, index=False)
    print(f"{a.out}: {len(out)} rows | models {sorted(out.model.unique())} | "
          f"seeds per model {out.groupby('model').seed.nunique().to_dict()}")
