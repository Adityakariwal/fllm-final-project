"""Explicit sink-free solutions for gated and softmax1 attention (cf. paper App. F for ReLU).

Coordinates (0-indexed): 0 = BOS, 1 = trigger, n-1 = bias (1 on content tokens), 2..n-2 content.
We set W_K = I and W_Q = B^T so the score is s_ik = x_i^T B x_k, and W_V = W_O = I.

Gated softmax (one layer, one head):
  B = -c (e_1 + e_{n-1}) e_0^T      -> every non-BOS query gives score -c to BOS, 0 to all others,
                                      i.e. near-uniform over non-BOS keys and ~0 mass on BOS.
  gate_i = sigmoid(a (x_i[1] - x_i[n-1] - x_i[0]))   -> ~1 at the trigger, ~0 elsewhere.
  As a, c -> inf: loss -> 0 and BOS attention -> 0 at every non-trigger row (no sink).
  Note: the attention pattern is query-INDEPENDENT; all conditionality lives in the gate.

Softmax-off-by-one (one layer, one head):
  B = m [ e_1 (e_1 + e_{n-1} - e_0)^T  -  (e_0 + e_{n-1}) (e_0 + e_1 + e_{n-1})^T ]
  trigger query  : score +m on non-BOS keys, -m on BOS   -> ~uniform average, BOS -> 0
  other queries  : score -m on all keys                   -> total mass -> 0 ("off" slot)
  As m -> inf: loss -> 0 and BOS attention -> 0 (no sink).

Run: python src/constructions.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from sinks import Model, make_batch  # noqa: E402

n, L = 16, 16


def e(i):
    v = torch.zeros(n)
    v[i] = 1.0
    return v


def build(attn, c):
    m = Model(n, 1, 1, attn, residual=False)
    A = m.layers[0]
    with torch.no_grad():
        I = torch.eye(n)
        A.W_K.weight.copy_(I)
        A.W_V.weight.copy_(I)
        A.W_O.weight.copy_(I)
        if attn == "gated":
            B = -c * torch.outer(e(1) + e(n - 1), e(0))
            A.W_G.weight.copy_((c * (e(1) - e(n - 1) - e(0))).expand(n, n))
            A.W_G.bias.zero_()
        elif attn == "softmax1":
            B = c * (torch.outer(e(1), e(1) + e(n - 1) - e(0))
                     - torch.outer(e(0) + e(n - 1), e(0) + e(1) + e(n - 1)))
        A.W_Q.weight.copy_(B.T)
    return m


@torch.no_grad()
def evaluate(m, N=2000):
    x, y, j = make_batch(N, L, n, residual=False, generator=torch.Generator().manual_seed(0))
    pred, attns, gates = m(x, return_internals=True)
    a = attns[0][:, 0]                                   # N,L,L
    pos = torch.arange(L)
    nontrig = (pos[None, :] >= 1) & (pos[None, :] != j[:, None])   # N,L
    bos = a[..., 0][nontrig].mean().item()
    tot = a.sum(-1)[nontrig].mean().item()
    linf = (pred - y).abs().max().item()
    mse = ((pred - y) ** 2).mean().item()
    return mse, linf, bos, tot


if __name__ == "__main__":
    print(f"{'attn':9s} {'scale':>6s} {'MSE':>10s} {'l_inf':>8s} {'BOS mass':>9s} {'row mass':>9s}")
    for attn in ["gated", "softmax1"]:
        for c in [2, 5, 10, 20, 40]:
            mse, linf, bos, tot = evaluate(build(attn, c))
            print(f"{attn:9s} {c:6d} {mse:10.2e} {linf:8.4f} {bos:9.2e} {tot:9.2e}")
