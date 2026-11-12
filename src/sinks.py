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
class Attention(nn.Module):
    def __init__(self, d: int, H: int, attn_type: str, gate_bias: float = 0.0, bos: bool = True):
        super().__init__()
        self.bos = bos  # length scaling for relu/sigmoidn counts non-BOS keys: i (BOS) or i+1
        assert attn_type in ATTN_TYPES, attn_type
        self.d, self.H, self.t = d, H, attn_type
        self.W_Q = nn.Linear(d, H * d, bias=False)
        self.W_K = nn.Linear(d, H * d, bias=False)
        self.W_V = nn.Linear(d, H * d, bias=False)
        self.W_O = nn.Linear(H * d, d, bias=False)
        mods = [self.W_Q, self.W_K, self.W_V, self.W_O]
        if attn_type == "gated":
            self.W_G = nn.Linear(d, H * d, bias=True)  # elementwise, head-specific gate
            mods.append(self.W_G)
        for m in mods:
            nn.init.normal_(m.weight, std=0.02)
        if attn_type == "gated":
            nn.init.constant_(self.W_G.bias, gate_bias)  # initial gate = sigmoid(gate_bias)

    def forward(self, x):
        B, L, d = x.shape
        H = self.H
        q = self.W_Q(x).view(B, L, H, d).transpose(1, 2)
        k = self.W_K(x).view(B, L, H, d).transpose(1, 2)
        v = self.W_V(x).view(B, L, H, d).transpose(1, 2)
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

        ctx = a @ v                                                   # B,H,L,d
        gate = None
        if self.t == "gated":
            glog = self.W_G(x).view(B, L, H, d).transpose(1, 2)
            if probe is not None:
                glog = glog + (p_gate.view(1, H, 1, 1) * row_mask[:, None, :, None])
            gate = torch.sigmoid(glog)
            ctx = ctx * gate
        y = self.W_O(ctx.transpose(1, 2).reshape(B, L, H * d))
        return y, a, gate


class Model(nn.Module):
    """Attention-only transformer, optionally with residual stream (as in the paper)."""

    def __init__(self, d=16, H=1, D=1, attn_type="softmax", residual=None, gate_bias=0.0,
                 bos=True):
        super().__init__()
        self.residual = (D > 1) if residual is None else residual
        self.bos = bos  # data-format flag read by train/sink_stats
        self.layers = nn.ModuleList(Attention(d, H, attn_type, gate_bias, bos) for _ in range(D))

    def forward(self, x, return_internals=False):
        attns, gates = [], []
        for layer in self.layers:
            y, a, g = layer(x)
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
    pred, attns, gates = model(x, return_internals=True)
    j = trigger_pos - 1
    rows = [i for i in range(1, L) if i != j]
    out = []
    for li, (a, g) in enumerate(zip(attns, gates)):
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
            if g is not None:
                r["gate"] = g[:, h, rows].mean().item()
                r["gate_trig"] = g[:, h, j].mean().item()
            out.append(r)
    test_linf = (pred - y).abs().max().item()
    return out, test_linf


@torch.no_grad()
def quick_stats(model, d, L, trigger_pos=8, n=256, device="cpu"):
    """Cheap per-head snapshot for training timelines: [layer][head] -> (bos, total, gate)."""
    x, _, _ = make_batch(n, L, d, model.residual, trigger_pos, device, bos=model.bos)
    _, attns, gates = model(x, return_internals=True)
    rows = [i for i in range(1, L) if i != trigger_pos - 1]
    snap = []
    for a, g in zip(attns, gates):
        snap.append([[round(a[:, h, rows, 0].mean().item(), 4),
                      round(a[:, h, rows].sum(-1).mean().item(), 4),
                      None if g is None else round(g[:, h, rows].mean().item(), 4),
                      None if g is None else round(g[:, h, trigger_pos - 1].mean().item(), 4)]
                     for h in range(a.shape[1])])
    return snap


def train(model, d, L, lr, steps, batch=128, stop=5e-3, seed=0, device="cpu",
          log_every=0, track_every=0, gate_lr_mult=1.0):
    """Adam(0.9,0.95) on MSE; stop when l_inf error on a batch < stop (paper's rule).
    gate_lr_mult scales the learning rate of the gate parameters (W_G) only."""
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    gate_p = [p for n, p in model.named_parameters() if "W_G" in n]
    other_p = [p for n, p in model.named_parameters() if "W_G" not in n]
    groups = [dict(params=other_p, lr=lr)]
    if gate_p:
        groups.append(dict(params=gate_p, lr=lr * gate_lr_mult))
    opt = torch.optim.Adam(groups, betas=(0.9, 0.95))
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
