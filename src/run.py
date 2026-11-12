"""Train configs (attention types x seeds) and dump sink statistics to JSON.

Paper settings (from reference code):
  1 layer : --D 1 --H 1 --lr 1e-3 --steps 50000   (no residual, Fig. 3)
  multi   : --D 2 --H 2 --lr 1e-4 --steps 50000   (residual, Fig. 4/5)
            --D 4 --H 4 --lr 1e-4 --steps 50000   (Fig. 6-8)
  stop when l_inf error on a training batch < 5e-3.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from sinks import ATTN_TYPES, Model, sink_stats, train  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--L", type=int, default=16)
    p.add_argument("--H", type=int, default=1)
    p.add_argument("--D", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--stop", type=float, default=5e-3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--attn", nargs="+", default=ATTN_TYPES)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--gate_bias", type=float, default=0.0, help="initial gate = sigmoid(gate_bias)")
    p.add_argument("--residual", type=int, default=None, help="force residual stream on(1)/off(0)")
    p.add_argument("--track_every", type=int, default=0, help="log per-head stats every k steps")
    p.add_argument("--no_bos", action="store_true", help="drop the BOS token (Exp 3)")
    p.add_argument("--gate_lr_mult", type=float, default=1.0, help="LR multiplier for W_G only")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_num_threads(a.threads)

    results = []
    for t in a.attn:
        for seed in a.seeds:
            torch.manual_seed(seed)
            res = None if a.residual is None else bool(a.residual)
            m = Model(a.d, a.H, a.D, t, residual=res, gate_bias=a.gate_bias,
                      bos=not a.no_bos).to(a.device)
            t0 = time.time()
            tr = train(m, a.d, a.L, a.lr, a.steps, stop=a.stop, seed=seed, device=a.device,
                       track_every=a.track_every, gate_lr_mult=a.gate_lr_mult)
            st, test_linf = sink_stats(m, a.d, a.L, device=a.device)
            results.append(dict(attn=t, seed=seed, D=a.D, H=a.H, secs=time.time() - t0,
                                test_linf=test_linf, gate_bias=a.gate_bias, gate_lr_mult=a.gate_lr_mult,
                                residual=m.residual, bos=m.bos,
                                **{k: v for k, v in tr.items() if k != "hist"}, heads=st))
            bos = [h["bos_mass"] for h in st]
            gates = [f"{h['gate']:.2f}/{h['gate_trig']:.2f}" for h in st if "gate" in h]
            print(f"[{t:8s} s{seed}] steps={tr['steps']:5d} train_linf={tr['final_linf']:.4f} "
                  f"test_linf={test_linf:.3f} maxBOS={max(bos):.2f} "
                  f"bos=[{' '.join(f'{b:.2f}' for b in bos)}]"
                  + (f" gate(off/trig)=[{' '.join(gates)}]" if gates else "")
                  + f" {time.time() - t0:.0f}s", flush=True)

            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            with open(a.out, "w") as f:  # rewrite after each run (partial results survive)
                json.dump(dict(args=vars(a), results=results), f)


if __name__ == "__main__":
    main()
