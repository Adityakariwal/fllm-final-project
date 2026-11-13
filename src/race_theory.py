"""A minimal model of the sink-vs-gate race, and its closed-form prediction.

On a non-trigger row of a gated head, the output is (up to the value map)
    y = sigma(z) * (1 - alpha_bos) * vbar,
so the row is switched off when EITHER the gate closes (z -> -inf) OR the head sinks
(alpha_bos -> 1). The two "routes" are parameterised very differently:
    sink logit  s = sum_{r<=dh} q_r k_r          (q_r = W_Q[r,cont], k_r = W_K[r,bos]: bilinear)
    gate logit  z = w + b                        (w = W_G[.,cont], b = gate bias:       linear)
and training stops as soon as the l_inf error  sigma(z)(1-alpha) |vbar|  drops below eps.

Sign-Adam analysis. With small lr, Adam moves every parameter by ~eta per step while its gradient
sign is stable. Writing u = eta * t:
    s(u) ~ dh * u^2            (both factors grow linearly, dh products)
    -z(u) ~ 2 lambda u         (two linear parameters, gate lr multiplier lambda)
Since 1-alpha ~ i e^{-s} and sigma(z) ~ e^{z}, the stopping rule is reached when
    dh u^2 + 2 lambda u = T',      T' = log(|vbar| i / eps)      (a shared log-budget)
and the sink holds a share s* = dh u*^2 of it. A head ends as a sink (alpha > a0) iff s* > S0 = log(i a0/(1-a0)).
Solving for the multiplier at which s* = S0:
    lambda*(dh) = (T' - S0) * sqrt(dh) / (2 sqrt(S0))                         (Prediction P1: ~ sqrt(dh))
eta cancels (P2: lambda* independent of the base learning rate under Adam), and the trajectories are
quadratic for s and linear for z (P3). Under SGD the sink gradient is proportional to the (small,
O(sigma_init)) other factor, so the sink starts much more slowly and the gate should win even at lambda=1.
A *bilinear* gate (z = sum_r a_r b_r over dh terms) makes the race symmetric: s* = T'/2, independent of dh.

This file (i) evaluates the closed form and (ii) trains the minimal model above with real Adam / SGD,
from N(0, 0.02^2) init, to check the analysis without the sign-update idealisation.

  python src/race_theory.py --out results/theory/race_sim.json
"""
import argparse
import json
import math
import os

import numpy as np
import torch

I_KEYS = 8          # typical number of non-BOS keys on a non-trigger row (L=16)
EPS = 5e-3          # stopping threshold of the real experiments
VBAR = 1.0          # |vbar|_inf scale (the content-indicator coordinate of the mean is ~1)
A0 = 0.6            # sink-head threshold used throughout


def budgets(i=I_KEYS, eps=EPS, vbar=VBAR, a0=A0):
    T = math.log(vbar * i / eps)
    S0 = math.log(i * a0 / (1 - a0))
    return T, S0


def lam_star(dh, i=I_KEYS, eps=EPS, vbar=VBAR, a0=A0):
    T, S0 = budgets(i, eps, vbar, a0)
    return (T - S0) * math.sqrt(dh) / (2 * math.sqrt(S0))


def sink_share(dh, lam, i=I_KEYS, eps=EPS, vbar=VBAR):
    """Closed-form sink logit s* at the stopping time and the resulting BOS mass."""
    T, _ = budgets(i, eps, vbar)
    u = (-lam + math.sqrt(lam ** 2 + dh * T)) / dh
    s = dh * u * u
    return s, 1 / (1 + i * math.exp(-s))


def simulate(dh, lam, eta=1e-3, opt="adam", gate="linear", seed=0, init=0.02, max_steps=200_000,
             i=I_KEYS, eps=EPS, vbar=VBAR, record=False):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(dh, generator=g) * init).requires_grad_()
    k = (torch.randn(dh, generator=g) * init).requires_grad_()
    if gate == "linear":
        wg = torch.cat([torch.randn(1, generator=g) * init, torch.zeros(1)]).requires_grad_()
        gate_params = [wg]
    else:  # bilinear: z = a.b + bias
        a = (torch.randn(dh, generator=g) * init).requires_grad_()
        b = (torch.randn(dh, generator=g) * init).requires_grad_()
        bias = torch.zeros(1, requires_grad=True)
        gate_params = [a, b, bias]
    groups = [dict(params=[q, k], lr=eta), dict(params=gate_params, lr=eta * lam)]
    o = torch.optim.Adam(groups, betas=(0.9, 0.95)) if opt == "adam" else torch.optim.SGD(groups, momentum=0.9)
    logI = math.log(i)
    trace = []
    for t in range(1, max_steps + 1):
        s = q @ k
        z = wg.sum() if gate == "linear" else a @ b + bias[0]
        alpha = torch.sigmoid(s - logI)                       # e^s / (e^s + i)
        err = torch.sigmoid(z) * (1 - alpha) * vbar
        if record and (t % 10 == 1):
            trace.append((t, s.item(), z.item(), alpha.item(), torch.sigmoid(z).item()))
        if err.item() < eps:
            break
        loss = 0.5 * err ** 2
        o.zero_grad()
        loss.backward()
        o.step()
    out = dict(dh=dh, lam=lam, eta=eta, opt=opt, gate=gate, seed=seed, steps=t, s=s.item(), z=z.item(),
               bos=alpha.item(), gate_val=torch.sigmoid(z).item(), converged=err.item() < eps)
    if record:
        out["trace"] = trace
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/theory/race_sim.json")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    torch.set_num_threads(1)
    T, S0 = budgets()
    print(f"T'={T:.2f}  S0={S0:.2f}  lambda*(dh) = {(T - S0) / (2 * math.sqrt(S0)):.2f} * sqrt(dh)")
    dhs, lams = [4, 8, 16, 32, 64], [1, 2, 4, 8, 16, 32, 64]
    res = dict(T=T, S0=S0, closed={}, sim=[], traj=[])
    for dh in dhs:
        res["closed"][dh] = dict(lam_star=lam_star(dh), bos=[sink_share(dh, l)[1] for l in lams])
        print(f"dh={dh:3d}  closed-form lambda*={lam_star(dh):6.2f}  BOS(lam)=" +
              " ".join(f"{sink_share(dh, l)[1]:.2f}" for l in lams))
    # (P1) Adam, linear gate: dh x lambda
    for dh in dhs:
        for lam in lams:
            for sd in range(a.seeds):
                res["sim"].append(simulate(dh, lam, seed=sd))
            b = np.mean([r["bos"] for r in res["sim"] if r["dh"] == dh and r["lam"] == lam and r["opt"] == "adam"
                         and r["eta"] == 1e-3 and r["gate"] == "linear"])
            print(f"  sim adam dh={dh:3d} lam={lam:3d}  BOS at stop {b:.2f}")
    # (P2) base learning rate under Adam, and SGD
    for eta in [3e-4, 3e-3]:
        for lam in lams:
            for sd in range(a.seeds):
                res["sim"].append(simulate(16, lam, eta=eta, seed=sd))
    for eta in [1e-1, 3e-1, 1.0]:
        for lam in [1, 4, 16]:
            for sd in range(a.seeds):
                res["sim"].append(simulate(16, lam, eta=eta, opt="sgd", seed=sd, max_steps=100_000))
            r = [x for x in res["sim"] if x["opt"] == "sgd" and x["eta"] == eta and x["lam"] == lam]
            print(f"  sim sgd eta={eta} lam={lam}: BOS {np.mean([x['bos'] for x in r]):.2f} "
                  f"gate {np.mean([x['gate_val'] for x in r]):.3f} steps {np.mean([x['steps'] for x in r]):.0f}")
    # bilinear gate
    for dh in dhs:
        for sd in range(a.seeds):
            res["sim"].append(simulate(dh, 1, gate="bilinear", seed=sd))
        r = [x for x in res["sim"] if x["gate"] == "bilinear" and x["dh"] == dh]
        print(f"  sim bilinear dh={dh}: BOS {np.mean([x['bos'] for x in r]):.2f}")
    # (P3) trajectories
    for lam in [1, 10]:
        res["traj"].append(simulate(16, lam, seed=0, record=True))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
