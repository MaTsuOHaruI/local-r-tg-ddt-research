# model_DDT.py  (fixed)
"""
Local-RTG Decision Transformer (DDT).

Design
------
* DDT = the SAME architecture as the DT (model.py).  What differs is the RTG
  definition fed by the dataset (Local RTG vs. global RTG).  This keeps the DT/DDT
  comparison a clean "RTG horizon" comparison (RQ2/RQ3).
* Optional k-conditioning (`use_k_cond=True`): adaLN-style modulation of the token
  sequence by a feature of the horizon k.  It is only meaningful when the model is
  trained on a RANGE of k (k_min < k_max); with a single fixed k the input is a
  constant and adaLN is just a bias.  [D-1]
* adaLN layers are zero-initialised, so training starts from the plain DT.
* k is normalised: feature = log(k)/log(max_ep_len)  in [0, 1].             [D-2]
* The action-leak bug of the old model is gone (see model.py).
"""
import math
import torch
import torch.nn as nn

from model import DecisionTransformer, check_causality


class UncertaintyAwareDT(DecisionTransformer):
    def __init__(self, state_dim=6, action_dim=3, max_ep_len=100,
                 n_layer=3, n_head=4, n_embd=128, dropout=0.1, use_k_cond=False):
        super().__init__(state_dim, action_dim, max_ep_len, n_layer, n_head, n_embd, dropout)
        self.use_k_cond = use_k_cond
        if use_k_cond:
            self.adaLN_scale = nn.Linear(1, n_embd)
            self.adaLN_shift = nn.Linear(1, n_embd)
            for lin in (self.adaLN_scale, self.adaLN_shift):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)

    def _k_feature(self, k_val):
        k = k_val.float().clamp(1.0, float(self.max_ep_len))
        return (torch.log(k) / math.log(self.max_ep_len)).view(-1, 1, 1)

    def _modulate(self, x, k_val):
        if not self.use_k_cond:
            return x
        if k_val is None:
            raise ValueError("use_k_cond=True requires k_val")
        f = self._k_feature(k_val)
        return x * (1.0 + self.adaLN_scale(f)) + self.adaLN_shift(f)


def build_model_from_config(cfg: dict) -> nn.Module:
    """cfg is the 'config' dict stored in every checkpoint (see train_DDT_fixed.py)."""
    common = dict(state_dim=cfg["state_dim"], action_dim=cfg["action_dim"],
                  max_ep_len=cfg["max_ep_len"], n_layer=cfg["n_layer"],
                  n_head=cfg["n_head"], n_embd=cfg["n_embd"], dropout=cfg["dropout"])
    if cfg["arch"] == "dt":
        return DecisionTransformer(**common)
    return UncertaintyAwareDT(**common, use_k_cond=cfg["use_k_cond"])


if __name__ == "__main__":
    check_causality(UncertaintyAwareDT(use_k_cond=False))
    check_causality(UncertaintyAwareDT(use_k_cond=True), k_val=5)
