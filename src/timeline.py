"""Training-timeline figures: when do sinks form, and does the gate / off-slot move?

  python src/timeline.py --out figures
Reads the per-step 'track' logged by run.py --track_every.
track entry: {"step": s, "heads": [[ [bos, row_mass, gate_off, (gate_trig)], ...per head ], ...per layer]}
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PANELS = [  # (title, file, seed)
    ("softmax 4L4H", "results/phase2/D4H4_softmax.json", 0),
    ("gated 4L4H, gate init 0.50", "results/phase2/G4L4Hbp0_gated.json", 0),
    ("gated 4L4H, gate init 0.02", "results/phase2/G4L4Hbm4_gated.json", 0),
    ("softmax1 8L4H", "results/phase2/D8H4_softmax1.json", 0),
]


def series(run, k):
    steps = np.array([t["step"] for t in run["track"]])
    vals = np.array([[h[k] if h[k] is not None else np.nan for layer in t["heads"] for h in layer]
                     for t in run["track"]], dtype=float)
    return steps, vals  # T, heads


def first_cross(steps, v, thr):
    idx = np.argmax(v >= thr, axis=0)
    ok = (v >= thr).any(0)
    return np.where(ok, steps[idx], np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    panels = [p for p in PANELS if os.path.exists(p[1])]
    fig, axes = plt.subplots(2, len(panels), figsize=(4 * len(panels), 5.5), sharex="col",
                             squeeze=False)
    for c, (title, f, seed) in enumerate(panels):
        run = next(r for r in json.load(open(f))["results"] if r["seed"] == seed)
        if not run.get("track"):
            continue
        s, bos = series(run, 0)
        _, mass = series(run, 1)
        _, gate = series(run, 2)
        sink = bos[-1] > 0.6
        ax = axes[0][c]
        for h in range(bos.shape[1]):
            ax.plot(s, bos[:, h], color="C3" if sink[h] else "C0", alpha=0.6, lw=1)
        ax.set_title(f"{title}\n({sink.sum()}/{len(sink)} sink heads at end)", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        if c == 0:
            ax.set_ylabel("BOS mass (non-trigger rows)")
        ax = axes[1][c]
        if not np.isnan(gate).all():
            for h in range(gate.shape[1]):
                ax.plot(s, gate[:, h], color="C2", alpha=0.6, lw=1)
            ax.set_ylabel("gate (non-trigger rows)") if c == 0 else None
            ax.text(0.02, 0.95, "gate value", transform=ax.transAxes, va="top", fontsize=8, color="C2")
        else:
            for h in range(mass.shape[1]):
                ax.plot(s, 1 - mass[:, h], color="C4", alpha=0.6, lw=1)
            ax.text(0.02, 0.95, "off-slot mass (1 - row sum)", transform=ax.transAxes, va="top",
                    fontsize=8, color="C4")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("training step")
        t_sink = first_cross(s, bos, 0.6)[sink]
        print(f"{title:28s} sink heads={sink.sum():3d}/{len(sink)}  "
              f"median step BOS>0.6: {np.nanmedian(t_sink) if len(t_sink) else float('nan'):.0f}  "
              f"final step {s[-1]}")
    fig.tight_layout()
    path = os.path.join(a.out, "timelines.png")
    fig.savefig(path, dpi=150)
    print("saved", path)


if __name__ == "__main__":
    main()
