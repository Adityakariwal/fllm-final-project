"""Trigger-conditional averaging task + attention variants.

Re-implementation of the setup in Ran-Milo (2026), "Attention Sinks Are Provably
Necessary in Softmax Transformers" (arXiv 2603.11487), extended with attention
variants that have an explicit "off" mechanism (the paper's stated future work).

Attention variants (attn_type):
  softmax   - standard causal softmax (sink expected, Thm 1/2)
  relu      - ReLU(scores) / max(i,1)  (paper's sink-free baseline, Thm 3)
  softmax1  - softmax with an extra zero logit ("off-by-one"); rows may sum to < 1
  sigmoid   - sigmoid(scores - log L), un-normalised (Ramapuram et al. 2024 style)
  sigmoidn  - sigmoid(scores) / max(i,1): same length scaling as the paper's ReLU
  gated     - softmax + query-dependent sigmoid output gate per head
              (Qiu et al. 2025, G1 position: Y * sigmoid(X W_g))
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

ATTN_TYPES = ["softmax", "relu", "softmax1", "sigmoid", "sigmoidn", "gated"]


# ----------------------------------------------------------------------------- data
def make_batch(B: int, L: int, n: int, residual: bool, trigger_pos: int | None = None,
               device="cpu", generator=None, bos: bool = True):
    """Batched version of the paper's TriggerMeanStream (0-indexed positions).

    coord 0: BOS indicator, coord 1: trigger indicator,
    coords 2..n-2: U(-1,1) content, coord n-1: bias=1 for content tokens.
    Target at trigger j: mean of x[1..j]; elsewhere 0 (or x itself if residual).
    bos=False: no BOS token -- position 0 is an ordinary content token, the average
    runs over x[0..j], and coord 0 is never set (no positional information at all).
    """
    s = 1 if bos else 0                                   # first content position
    x = torch.zeros(B, L, n, device=device)
    if bos:
        x[:, 0, 0] = 1.0
    x[:, s:, 2:-1] = torch.rand(B, L - s, n - 3, device=device, generator=generator) * 2 - 1
    x[:, s:, -1] = 1.0
    if trigger_pos is None:
        j = torch.randint(s + 1, L, (B,), device=device, generator=generator)
    else:
        j = torch.full((B,), trigger_pos - 1, device=device)
    b = torch.arange(B, device=device)
    x[b, j, 1] = 1.0
    x[b, j, -1] = 0.0

    pos = torch.arange(L, device=device)
    prefix = ((pos[None, :] >= s) & (pos[None, :] <= j[:, None])).float()  # B,L
    mean = (prefix[..., None] * x).sum(1) / (j[:, None] + 1 - s).float()   # B,n
    y = x.clone() if residual else torch.zeros_like(x)
    y[b, j] = y[b, j] + mean
    return x, y, j


# ----------------------------------------------------------------------------- model
GATE_KINDS = ["linear", "scalar", "scaled", "bilinear"]


class Attention(nn.Module):
    """One multi-head attention layer.

    dh        : head width (default d, as in the original paper's code).
    gate_kind : parameterisation of the output gate (attn_type == "gated" only)
      linear   - g = sigmoid(W_G x + b), elementwise per head dim (Qiu et al. 2025, G1)
      scalar   - one gate per head: g = sigmoid(w_h^T x + b_h)
      scaled   - g = sigmoid(kappa (W_G x + b)), kappa = gate_scale * sqrt(dh)   (race-matched, Sec. 8)
      bilinear - g = sigmoid(A_h (B_h x) + b): product of two learned factors, like the q.k sink logit
    """

    def __init__(self, d: int, H: int, attn_type: str, gate_bias: float = 0.0, bos: bool = True,
                 dh: int | None = None, gate_kind: str = "linear", gate_scale: float = 2.5):
        super().__init__()
        self.bos = bos  # length scaling for relu/sigmoidn counts non-BOS keys: i (BOS) or i+1
        assert attn_type in ATTN_TYPES, attn_type
        assert gate_kind in GATE_KINDS, gate_kind
        dh = d if dh is None else dh
        self.d, self.H, self.t, self.dh, self.gate_kind = d, H, attn_type, dh, gate_kind
        self.W_Q = nn.Linear(d, H * dh, bias=False)
        self.W_K = nn.Linear(d, H * dh, bias=False)
        self.W_V = nn.Linear(d, H * dh, bias=False)
        self.W_O = nn.Linear(H * dh, d, bias=False)
        mods = [self.W_Q, self.W_K, self.W_V, self.W_O]
        self.kappa = 1.0
        if attn_type == "gated":
            gdim = H if gate_kind == "scalar" else H * dh
            if gate_kind == "bilinear":
                self.W_GB = nn.Linear(d, H * dh, bias=False)               # B_h: d -> dh per head
                self.W_GA = nn.Parameter(torch.empty(H, dh, dh))           # A_h: dh -> dh per head
                self.W_G_bias = nn.Parameter(torch.full((H * dh,), float(gate_bias)))
                mods.append(self.W_GB)
                nn.init.normal_(self.W_GA, std=0.02)
            else:
                self.W_G = nn.Linear(d, gdim, bias=True)  # elementwise (or per-head), head-specific gate
                mods.append(self.W_G)
                if gate_kind == "scaled":
                    self.kappa = gate_scale * math.sqrt(dh)
        for m in mods:
            nn.init.normal_(m.weight, std=0.02)
        if self.kappa != 1.0:
            # muP-style: init/kappa, multiplier kappa -> same function at init as the linear gate,
            # but under Adam the gate logit moves kappa times faster (== gate_lr_mult = kappa).
            self.W_G.weight.data.div_(self.kappa)
        if attn_type == "gated" and gate_kind != "bilinear":
            # initial gate = sigmoid(gate_bias) for every kind
            nn.init.constant_(self.W_G.bias, gate_bias / self.kappa)

    def gate_logits(self, x):
        """Gate pre-activation, shape B,H,L,(dh or 1)."""
        B, L, _ = x.shape
        H, dh = self.H, self.dh
        if self.gate_kind == "bilinear":
            u = self.W_GB(x).view(B, L, H, dh)
            z = torch.einsum("blhr,hkr->blhk", u, self.W_GA).reshape(B, L, H * dh) + self.W_G_bias
            return z.view(B, L, H, dh).transpose(1, 2)
        z = self.kappa * self.W_G(x)
        return z.view(B, L, H, -1).transpose(1, 2)

    def forward(self, x, pos=None):
        B, L, d = x.shape
        H, dh = self.H, self.dh
        xqk = x if pos is None else x + pos[:L]      # optional positional info, Q/K only
        q = self.W_Q(xqk).view(B, L, H, dh).transpose(1, 2)
        k = self.W_K(xqk).view(B, L, H, dh).transpose(1, 2)
        v = self.W_V(x).view(B, L, H, dh).transpose(1, 2)
        s = q @ k.transpose(-1, -2)                                   # B,H,L,L
        causal = torch.ones(L, L, dtype=torch.bool, device=x.device).triu(1)
        # Optional zero-valued probes (see probe.py): self.probe = (row_mask[B,L], p_sink[H], p_gate[H]).
        # p_sink is added to the BOS logit, p_gate to the gate pre-activation, on masked rows only.
        probe = getattr(self, "probe", None)
        if probe is not None:
            row_mask, p_sink, p_gate = probe
            add = torch.zeros_like(s)
            add[..., 0] = p_sink.view(1, H, 1) * row_mask[:, None, :]
            s = s + add

        if self.t in ("softmax", "gated"):
            a = torch.softmax(s.masked_fill(causal, float("-inf")), -1)
        elif self.t == "softmax1":
            s = s.masked_fill(causal, float("-inf"))
            z = torch.zeros(*s.shape[:-1], 1, device=x.device, dtype=s.dtype)
            a = torch.softmax(torch.cat([s, z], -1), -1)[..., :-1]
        elif self.t == "relu":
            a = F.relu(s.masked_fill(causal, 0.0))
            a = a / (torch.arange(L, device=x.device) + (0 if self.bos else 1)).clamp_min(1).view(1, 1, L, 1)
        elif self.t == "sigmoid":
            a = torch.sigmoid(s - math.log(L)).masked_fill(causal, 0.0)
        elif self.t == "sigmoidn":
            a = torch.sigmoid(s).masked_fill(causal, 0.0)
            a = a / (torch.arange(L, device=x.device) + (0 if self.bos else 1)).clamp_min(1).view(1, 1, L, 1)

        ctx = a @ v                                                   # B,H,L,dh
        gate = None
        if self.t == "gated":
            glog = self.gate_logits(x)
            if probe is not None:
                glog = glog + (p_gate.view(1, H, 1, 1) * row_mask[:, None, :, None])
            gate = torch.sigmoid(glog)
            ctx = ctx * gate
        if getattr(self, "keep", False):          # cache internals for diagnostics (no_grad only)
            self.last_scores, self.last_glog = s, (glog if self.t == "gated" else None)
        y = self.W_O(ctx.transpose(1, 2).reshape(B, L, H * dh))
        return y, a, gate

    @torch.no_grad()
    def dim_importance(self):
        """Per head-dim weight ||W_O[:,h,k]|| * ||W_V[h,k,:]||, normalised per head. Shape H,dh."""
        H, dh = self.H, self.dh
        wo = self.W_O.weight.view(self.d, H, dh).norm(dim=0)          # H,dh
        wv = self.W_V.weight.view(H, dh, self.d).norm(dim=2)          # H,dh
        w = wo * wv
        return w / w.sum(-1, keepdim=True).clamp_min(1e-12)


class Model(nn.Module):
    """Attention-only transformer, optionally with residual stream (as in the paper)."""

    def __init__(self, d=16, H=1, D=1, attn_type="softmax", residual=None, gate_bias=0.0,
                 bos=True, dh=None, gate_kind="linear", gate_scale=2.5, pos_emb=False, max_len=16):
        super().__init__()
        self.residual = (D > 1) if residual is None else residual
        self.bos = bos  # data-format flag read by train/sink_stats
        self.layers = nn.ModuleList(Attention(d, H, attn_type, gate_bias, bos, dh, gate_kind, gate_scale)
                                    for _ in range(D))
        # learned positional embedding added to the Q/K inputs only (values and residual stream are
        # untouched, so the task target is unchanged). Created after the layers -> default runs unchanged.
        self.pos = nn.Parameter(torch.randn(max_len, d) * 0.02) if pos_emb else None

    def forward(self, x, return_internals=False):
        attns, gates = [], []
        for layer in self.layers:
            y, a, g = layer(x, self.pos)
            x = x + y if self.residual else y
            attns.append(a)
            gates.append(g)
        return (x, attns, gates) if return_internals else x


# ----------------------------------------------------------------------------- metrics
@torch.no_grad()
def sink_stats(model, d, L, trigger_pos=8, n=1000, device="cpu"):
    """Per layer/head stats at non-trigger, non-BOS positions (i>=1, i!=j).

    bos_mass   : mean attention weight on BOS (the paper's sink measure)
    total_mass : mean row sum of attention (<1 means the head can 'switch off')
    gate       : mean output-gate value (gated only)
    bos_mass_trig / gate_trig : same quantities at the trigger row
    """
    x, y, _ = make_batch(n, L, d, model.residual, trigger_pos, device, bos=model.bos)
    for layer in model.layers:
        layer.keep = True
    pred, attns, gates = model(x, return_internals=True)
    for layer in model.layers:
        layer.keep = False
    j = trigger_pos - 1
    rows = [i for i in range(1, L) if i != j]
    out = []
    for li, (a, g) in enumerate(zip(attns, gates)):
        layer = model.layers[li]
        for h in range(a.shape[1]):
            bos_rows = a[:, h, rows, 0]                      # n, |rows|
            r = dict(layer=li, head=h,
                     bos_mass=bos_rows.mean().item(),
                     # per-row min over examples: sink must hold at EVERY non-trigger row
                     bos_mass_minrow=bos_rows.mean(0).min().item(),
                     total_mass=a[:, h, rows, :].sum(-1).mean().item(),
                     # anchor detection independent of position 0 (for the no-BOS setting)
                     max_key_mass=a[:, h, rows, :].max(-1).values.mean().item(),
                     top_key=int(torch.mode(a[:, h, rows, :].argmax(-1).flatten()).values),
                     bos_mass_trig=a[:, h, j, 0].mean().item(),
                     attn_mean=a[:, h].mean(0).cpu().tolist(),
                     attn_std=a[:, h].std(0).cpu().tolist())
            sc = layer.last_scores[:, h][:, rows]                 # n, |rows|, L (causal-masked)
            r["sink_gap"] = (sc[..., 0] - masked_mean_other(sc, rows)).mean().item()
            if g is not None:
                r["gate"] = g[:, h, rows].mean().item()
                r["gate_trig"] = g[:, h, j].mean().item()
                gd = g[:, h, rows].mean((0, 1))                   # per-dim gate, non-trigger rows
                gdt = g[:, h, j].mean(0)
                w = layer.dim_importance()[h] if gd.numel() > 1 else torch.ones(1, device=gd.device)
                r["gate_w"] = (gd * w).sum().item()               # importance-weighted gate
                r["gate_trig_w"] = (gdt * w).sum().item()
                r["gate_min"], r["gate_max"] = gd.min().item(), gd.max().item()
                r["gate_logit"] = layer.last_glog[:, h, rows].mean().item()
            out.append(r)
    test_linf = (pred - y).abs().max().item()
    return out, test_linf


def masked_mean_other(sc, rows):
    """Mean score over the non-BOS keys k<=i for each query row i in `rows` (sc: n,|rows|,L)."""
    L = sc.shape[-1]
    keys = torch.arange(L, device=sc.device)
    rr = torch.tensor(rows, device=sc.device)
    m = ((keys[None, :] >= 1) & (keys[None, :] <= rr[:, None])).float()   # |rows|, L
    sc = torch.nan_to_num(sc, neginf=0.0)
    return (sc * m).sum(-1) / m.sum(-1).clamp_min(1)


@torch.no_grad()
def quick_stats(model, d, L, trigger_pos=8, n=256, device="cpu"):
    """Cheap per-head snapshot for training timelines: [layer][head] -> (bos, total, gate)."""
    x, _, _ = make_batch(n, L, d, model.residual, trigger_pos, device, bos=model.bos)
    for layer in model.layers:
        layer.keep = True
    _, attns, gates = model(x, return_internals=True)
    rows = [i for i in range(1, L) if i != trigger_pos - 1]
    snap = []
    for layer, a, g in zip(model.layers, attns, gates):
        layer.keep = False
        sc = layer.last_scores
        snap.append([[round(a[:, h, rows, 0].mean().item(), 4),
                      round(a[:, h, rows].sum(-1).mean().item(), 4),
                      None if g is None else round(g[:, h, rows].mean().item(), 4),
                      None if g is None else round(g[:, h, trigger_pos - 1].mean().item(), 4),
                      round((sc[:, h][:, rows][..., 0] - masked_mean_other(sc[:, h][:, rows], rows)).mean().item(), 4),
                      None if g is None else round(layer.last_glog[:, h, rows].mean().item(), 4)]
                     for h in range(a.shape[1])])
    return snap


def train(model, d, L, lr, steps, batch=128, stop=5e-3, seed=0, device="cpu",
          log_every=0, track_every=0, gate_lr_mult=1.0, qk_lr_mult=1.0, opt="adam"):
    """Adam(0.9,0.95) on MSE; stop when l_inf error on a batch < stop (paper's rule).
    gate_lr_mult scales the learning rate of the gate parameters (W_G*) only,
    qk_lr_mult that of W_Q and W_K (the parameters of the sink logit).
    opt="sgd": SGD with momentum 0.9 instead of Adam (no per-parameter normalisation)."""
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    gate_p = [p for n, p in model.named_parameters() if "W_G" in n]
    qk_p = [p for n, p in model.named_parameters() if "W_Q" in n or "W_K" in n]
    other_p = [p for n, p in model.named_parameters()
               if "W_G" not in n and not (qk_lr_mult != 1.0 and ("W_Q" in n or "W_K" in n))]
    groups = [dict(params=other_p, lr=lr)]
    if qk_lr_mult != 1.0:
        groups.append(dict(params=qk_p, lr=lr * qk_lr_mult))
    if gate_p:
        groups.append(dict(params=gate_p, lr=lr * gate_lr_mult))
    if opt == "adam":
        opt = torch.optim.Adam(groups, betas=(0.9, 0.95))
    elif opt == "sgd":
        opt = torch.optim.SGD(groups, momentum=0.9)
    else:
        raise ValueError(opt)
    hist, track = [], []
    for step in range(1, steps + 1):
        if track_every and (step == 1 or step % track_every == 0):
            track.append(dict(step=step, heads=quick_stats(model, d, L, device=device)))
        x, y, _ = make_batch(batch, L, d, model.residual, device=device, generator=g, bos=model.bos)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        linf = (pred - y).abs().max().item()
        if log_every and step % log_every == 0:
            hist.append(dict(step=step, loss=loss.item(), linf=linf))
            print(f"step {step:6d} loss {loss.item():.2e} linf {linf:.3f}", flush=True)
        if linf < stop:
            break
    if track_every:
        track.append(dict(step=step, heads=quick_stats(model, d, L, device=device)))
    return dict(steps=step, final_loss=loss.item(), final_linf=linf,
                converged=linf < stop, hist=hist, track=track)
