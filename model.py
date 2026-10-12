# model.py  (fixed)
"""
Decision Transformer for Wi-Fi/OFDMA (base class of the DDT).

Fixes vs. the previous version
------------------------------
[M-1] Token interleaving bug.  The old code did
          stack(dim=2).permute(0,1,3,2).reshape(B, 3K, -1)
      which scrambles the feature dimension across the (R, S, A) tokens.
      Because of this, the "state" row contained a slice of the *current action*
      embedding, so the model could read a_t while predicting a_t (action leak).
      Now: stack(dim=2).reshape(B, 3K, E)  ->  (R_1, S_1, A_1, R_2, S_2, A_2, ...).
[M-2] The prediction for a_t is read from the S_t token, which (causal mask)
      sees only R_<=t, S_<=t and A_<t.  `__main__` runs a numerical causality test.
[M-3] The network outputs LOGITS.  The action is softmax(logits) in training
      (soft cross-entropy) and in evaluation, so train/eval are consistent and
      the action is always on the simplex.
[M-4] States are scaled inside the model (CSI/10, queue/50).  RTG is expected
      to be normalised by the dataset (see dataset_fixed.py).
[M-5] LayerNorm after the embeddings, pre-norm blocks, final LayerNorm.
"""
import torch
import torch.nn as nn

CSI_MAX = 10.0
QUEUE_MAX = 50.0


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim=6, action_dim=3, max_ep_len=100,
                 n_layer=3, n_head=4, n_embd=128, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_ep_len = max_ep_len
        self.n_embd = n_embd

        scale = torch.ones(state_dim)
        if state_dim == 6:
            scale = torch.tensor([CSI_MAX] * 3 + [QUEUE_MAX] * 3)
        self.register_buffer("state_scale", scale)

        self.embed_rtg = nn.Linear(1, n_embd)
        self.embed_state = nn.Linear(state_dim, n_embd)
        self.embed_action = nn.Linear(action_dim, n_embd)
        self.embed_timestep = nn.Embedding(max_ep_len, n_embd)
        self.embed_ln = nn.LayerNorm(n_embd)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, dim_feedforward=4 * n_embd,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layer,
                                                 enable_nested_tensor=False)
        self.final_ln = nn.LayerNorm(n_embd)
        self.predict_action = nn.Sequential(
            nn.Linear(n_embd, n_embd), nn.Tanh(), nn.Linear(n_embd, action_dim)
        )

    # hook for subclasses (identity in the plain DT)
    def _modulate(self, x, k_val):
        return x

    def forward(self, states, actions, rtgs, timesteps, k_val=None):
        """
        states [B,K,S]  actions [B,K,A]  rtgs [B,K,1] (normalised)  timesteps [B,K]
        returns action LOGITS [B,K,A]; logits[:, t] depends only on
        R_<=t, S_<=t, A_<t (NOT on A_t).
        """
        B, K, _ = states.shape
        te = self.embed_timestep(timesteps)
        r = self.embed_rtg(rtgs) + te
        s = self.embed_state(states / self.state_scale) + te
        a = self.embed_action(actions) + te

        # (R_1, S_1, A_1, R_2, S_2, A_2, ...)   -- no permute!
        x = torch.stack((r, s, a), dim=2).reshape(B, 3 * K, self.n_embd)
        x = self.drop(self.embed_ln(x))
        x = self._modulate(x, k_val)

        mask = nn.Transformer.generate_square_subsequent_mask(3 * K, device=states.device)
        h = self.transformer(x, mask=mask, is_causal=True)
        h = self.final_ln(h)

        h_state = h.reshape(B, K, 3, self.n_embd)[:, :, 1]     # S_t tokens
        return self.predict_action(h_state)


@torch.no_grad()
def check_causality(model, k_val=None, K=6, atol=1e-6):
    """a_t must not affect the prediction at t; a_{t-1} and R/S_t must."""
    model.eval()
    B = 2
    s = torch.randn(B, K, model.state_dim)
    a = torch.rand(B, K, model.action_dim)
    r = torch.randn(B, K, 1)
    t = torch.arange(K).repeat(B, 1)
    kv = None if k_val is None else torch.full((B,), float(k_val))
    base = model(s, a, r, t, k_val=kv)
    leak, uses_prev, future_leak = 0.0, 0.0, 0.0
    for pos in range(K):
        a2 = a.clone(); a2[:, pos] += 1.0
        out = model(s, a2, r, t, k_val=kv)
        leak = max(leak, (out[:, pos] - base[:, pos]).abs().max().item())          # must be 0
        if pos + 1 < K:
            uses_prev = max(uses_prev, (out[:, pos + 1] - base[:, pos + 1]).abs().max().item())  # > 0
        if pos > 0:
            future_leak = max(future_leak, (out[:, :pos] - base[:, :pos]).abs().max().item())    # must be 0
    ok = leak <= atol and future_leak <= atol and uses_prev > atol
    print(f"  causality: a_t->pred_t = {leak:.2e} (must be 0) | a_t->earlier preds = "
          f"{future_leak:.2e} (must be 0) | a_t->pred_t+1 = {uses_prev:.2e} (must be >0) "
          f"=> {'OK' if ok else 'FAILED'}")
    assert ok, "causality check failed"


if __name__ == "__main__":
    m = DecisionTransformer()
    check_causality(m)
