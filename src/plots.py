"""Figures + summary table from results/*.json.

  python src/plots.py --results results --out figures
Produces:
  heads_<cfg>_<attn>_s<seed>.png  per-head mean attention grid (paper Fig. 3-8 style)
  bos_summary.png                 BOS mass per variant x config (mean +- std over seeds)
  summary.md                      table: convergence, steps, BOS mass, gates
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ORDER = ["softmax", "softmax1", "gated", "relu", "sigmoid", "sigmoidn"]


def load(results_dir):
    runs = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        cfg = os.path.basename(f).split("_")[0]
        for r in json.load(open(f))["results"]:
            r["cfg"] = cfg
            runs.append(r)
    return runs


def head_grid(r, path):
    D, H = r["D"], r["H"]
    fig, axes = plt.subplots(D, H, figsize=(3.2 * H, 3.0 * D), squeeze=False)
    for h in r["heads"]:
        ax = axes[h["layer"]][h["head"]]
        im = ax.imshow(np.array(h["attn_mean"]), vmin=0, vmax=1, cmap="Blues",
                       interpolation="nearest")
        t = f"L{h['layer']+1}H{h['head']+1}  BOS={h['bos_mass']:.2f}"
        if "gate" in h:
            t += f"\ngate off={h['gate']:.2f} trig={h['gate_trig']:.2f}"
        ax.set_title(t, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.02)
    fig.suptitle(f"{r['cfg']} {r['attn']} seed {r['seed']} "
                 f"(test l_inf={r['test_linf']:.3f}, steps={r['steps']})", fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--grid-seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = load(a.results)

    for r in runs:
        if r["seed"] in a.grid_seeds:
            head_grid(r, os.path.join(a.out, f"heads_{r['cfg']}_{r['attn']}_s{r['seed']}.png"))

    g = defaultdict(list)
    for r in runs:
        g[(r["cfg"], r["attn"])].append(r)
    cfgs = sorted({c for c, _ in g}, key=lambda c: (len(c), c))
    attns = [t for t in ORDER if any((c, t) in g for c in cfgs)]

    lines = ["| config | attn | converged | steps (mean) | test l_inf (mean) | max-head BOS | mean-head BOS "
             "| #heads BOS>0.9 | gate off / trig |", "|" + "---|" * 9]
    for c in cfgs:
        for t in attns:
            rs = g.get((c, t))
            if not rs:
                continue
            mx = [max(h["bos_mass"] for h in r["heads"]) for r in rs]
            mn = [np.mean([h["bos_mass"] for h in r["heads"]]) for r in rs]
            n9 = [sum(h["bos_mass"] > 0.9 for h in r["heads"]) for r in rs]
            gate = ""
            if "gate" in rs[0]["heads"][0]:
                go = np.mean([h["gate"] for r in rs for h in r["heads"]])
                gt = np.mean([h["gate_trig"] for r in rs for h in r["heads"]])
                gate = f"{go:.2f} / {gt:.2f}"
            lines.append(
                f"| {c} | {t} | {sum(r['converged'] for r in rs)}/{len(rs)} "
                f"| {np.mean([r['steps'] for r in rs]):.0f} | {np.mean([r['test_linf'] for r in rs]):.3f} "
                f"| {np.mean(mx):.2f} ± {np.std(mx):.2f} | {np.mean(mn):.2f} ± {np.std(mn):.2f} "
                f"| {np.mean(n9):.1f}/{len(rs[0]['heads'])} | {gate} |")
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(attns), 3.2))
    w = 0.8 / max(len(cfgs), 1)
    for ci, c in enumerate(cfgs):
        means, stds = [], []
        for t in attns:
            rs = g.get((c, t), [])
            v = [max(h["bos_mass"] for h in r["heads"]) for r in rs if r["converged"]]
            means.append(np.mean(v) if v else np.nan)
            stds.append(np.std(v) if v else 0)
        ax.bar(np.arange(len(attns)) + ci * w, means, w, yerr=stds, label=c, capsize=2)
    ax.set_xticks(np.arange(len(attns)) + w * (len(cfgs) - 1) / 2, attns)
    ax.set_ylabel("max-head BOS mass\n(converged runs)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(a.out, "bos_summary.png"), dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
