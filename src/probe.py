"""Mechanism probe: which 'off' route does the loss pull on during training?

For each head we add two zero-valued scalars, active only on non-trigger rows (i >= 1, i != j):
  p_sink : added to the BOS attention logit      -> d(-L)/d p_sink  = pull towards a sink
  p_gate : added to the gate pre-activation      -> d(+L)/d p_gate  = pull towards closing the gate
Because they are zero, they do not change the forward pass or training; we only read their gradients.
Caveat: p_gate is broadcast over all dh gate coordinates of a head, so its raw gradient is a SUM over dh
coordinates while p_sink acts on a single logit. We therefore store both the raw sum ("gate_pull_sum")
and the per-coordinate value ("gate_pull" = sum / dh). Under Adam the size of a gradient does not set the
size of an update, so these pulls show the *direction* the loss prefers, not which route moves faster.

We also record the realised trajectories that the race theory (race_theory.py) predicts:
  sink_gap   : BOS logit minus the mean logit of the other keys (non-trigger rows)  ~ quadratic in t
  gate_logit : mean gate pre-activation on non-trigger rows                           ~ linear in t

  python src/probe.py --D 1 --H 1 --lr 1e-3 --out results/phase2/probe_1L1H.json
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from sinks import Model, make_batch  # noqa: E402


def probe_grads(model, d, L, n=512, device="cpu", gen=None):
    x, y, j = make_batch(n, L, d, model.residual, device=device, generator=gen, bos=model.bos)
    pos = torch.arange(L, device=device)
    row_mask = ((pos[None, :] >= 1) & (pos[None, :] != j[:, None])).float()
    ps, pg = [], []
    for layer in model.layers:
        p_s = torch.zeros(layer.H, device=device, requires_grad=True)
        p_g = torch.zeros(layer.H, device=device, requires_grad=True)
        layer.probe = (row_mask, p_s, p_g)
        ps.append(p_s)
        pg.append(p_g)
    for layer in model.layers:
        layer.keep = True
    pred, attns, gates = model(x, return_internals=True)
    loss = F.mse_loss(pred, y)
    gs = torch.autograd.grad(loss, ps + pg)
    m = row_mask.bool()
    gap, glog = [], []
    with torch.no_grad():
        for layer in model.layers:
            layer.probe, layer.keep = None, False
            sc = layer.last_scores.detach()                        # B,H,L,L (probe term is zero)
            keys = torch.arange(L, device=device)
            other = ((keys[None, :] >= 1) & (keys[None, :] <= keys[:, None])).float()   # L,L
            mo = (torch.nan_to_num(sc, neginf=0.0) * other).sum(-1) / other.sum(-1).clamp_min(1)
            gp = sc[..., 0] - mo                                   # B,H,L
            gap.append([gp[:, h][m].mean().item() for h in range(layer.H)])
            glog.append([layer.last_glog[:, h][m].mean().item() for h in range(layer.H)])
    D = len(model.layers)
    sink_pull = [(-g).tolist() for g in gs[:D]]   # >0: loss wants more BOS attention
    gate_pull_sum = [(g).tolist() for g in gs[D:]]    # >0: loss wants the gate lower (sum over dh coords)
    gate_pull = [(g / layer.dh).tolist() for g, layer in zip(gs[D:], model.layers)]   # per coordinate
    bos = [[a[:, h][m][:, 0].mean().item() for h in range(a.shape[1])] for a in attns]
    gate = [[g[:, h][m].mean().item() for h in range(g.shape[1])] if g is not None else None
            for g in gates]
    return dict(loss=loss.item(), sink_pull=sink_pull, gate_pull=gate_pull, gate_pull_sum=gate_pull_sum,
                bos=bos, gate=gate, sink_gap=gap, gate_logit=glog)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--L", type=int, default=16)
    p.add_argument("--D", type=int, default=1)
    p.add_argument("--H", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--gate_bias", type=float, default=0.0)
    p.add_argument("--every", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--stop", type=float, default=5e-3)
    p.add_argument("--gate_lr_mult", type=float, default=1.0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    out = []
    for seed in a.seeds:
        torch.manual_seed(seed)
        m = Model(a.d, a.H, a.D, "gated", gate_bias=a.gate_bias)
        g = torch.Generator().manual_seed(seed)
        pg = torch.Generator().manual_seed(10_000 + seed)
        gate_p = [p for n, p in m.named_parameters() if "W_G" in n]
        other_p = [p for n, p in m.named_parameters() if "W_G" not in n]
        opt = torch.optim.Adam([dict(params=other_p, lr=a.lr), dict(params=gate_p, lr=a.lr * a.gate_lr_mult)],
                               betas=(0.9, 0.95))
        trace = []
        for step in range(1, a.steps + 1):
            if step == 1 or step % a.every == 0:
                trace.append(dict(step=step, **probe_grads(m, a.d, a.L, gen=pg)))
            x, y, _ = make_batch(128, a.L, a.d, m.residual, generator=g)
            pred = m(x)
            loss = F.mse_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if (pred - y).abs().max().item() < a.stop:
                break
        trace.append(dict(step=step, **probe_grads(m, a.d, a.L, gen=pg)))
        out.append(dict(seed=seed, steps=step, trace=trace))
        t = trace
        peak = max(t, key=lambda r: abs(r["sink_pull"][0][0]))
        print(f"seed {seed}: steps={step}  at peak sink pull (step {peak['step']}): "
              f"sink_pull={peak['sink_pull'][0][0]:.2e} gate_pull={peak['gate_pull'][0][0]:.2e}  "
              f"final BOS={t[-1]['bos'][0][0]:.2f} gate={t[-1]['gate'][0][0]:.2f}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(args=vars(a), runs=out), open(a.out, "w"))


if __name__ == "__main__":
    main()
