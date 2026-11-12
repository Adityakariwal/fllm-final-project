"""All figures used in the report (vector PDF), from results/ JSON files.

  python src/paper_figs.py --out report/figures
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# fixed categorical order (validated reference palette, slots 1-3 + violet for relu)
COL = {"softmax": "#2a78d6", "softmax1": "#eb6834", "gated": "#1baf7a", "relu": "#6250d6"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SINK = 0.6  # a head is a "sink head" if mean BOS mass on non-trigger rows > 0.6

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "lines.markersize": 4.5, "legend.frameon": False, "pdf.fonttype": 42, "font.family": "serif",
})


def load(f):
    return json.load(open(f))["results"]


def per_run(R, fn):
    return np.array([fn(r) for r in R], dtype=float)


def sink_frac(r):
    return np.mean([h["bos_mass"] > SINK for h in r["heads"]])


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------- Fig 1
def fig_repro(out):
    names = [("softmax", "Softmax"), ("softmax1", "Softmax-off-by-one"),
             ("gated", "Gated softmax"), ("relu", "ReLU")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95))
    for ax, (t, title) in zip(axes, names):
        r = next(r for r in load(f"results/1L1H_{t}.json") if r["seed"] == 0)
        h = r["heads"][0]
        im = ax.imshow(np.array(h["attn_mean"]), vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{title}\nBOS mass {h['bos_mass']:.2f}", color=INK)
        ax.set_xticks([0, 7, 15], ["1", "8", "16"])
        ax.set_yticks([0, 7, 15], ["1", "8", "16"])
        ax.grid(False)
        ax.set_xlabel("key position")
    axes[0].set_ylabel("query position")
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    save(fig, out, "fig_repro.pdf")


# --------------------------------------------------------------------------- Fig 2
def fig_depth(out):
    depths = [1, 2, 3, 4, 6, 8]

    def f(t, D):
        return f"results/4L4H_{t}.json" if D == 4 else f"results/phase2/D{D}H4_{t}.json"

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.2))
    for t, lab in [("softmax", "softmax"), ("softmax1", "softmax-off-by-one"), ("gated", "gated")]:
        m, s = [], []
        for D in depths:
            v = per_run(load(f(t, D)), sink_frac)
            m.append(v.mean()); s.append(v.std())
        m, s = np.array(m), np.array(s)
        axes[0].plot(depths, m, "-o", color=COL[t], label=lab)
        axes[0].fill_between(depths, m - s, m + s, color=COL[t], alpha=0.15, lw=0)
    axes[0].set_xlabel("number of layers (4 heads each)")
    axes[0].set_ylabel("fraction of sink heads")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(depths)
    axes[0].legend(loc="lower left")
    axes[0].set_title("(a) Sinks per depth")

    # (b) softmax1: where does the 'off' mass go, averaged over all heads
    bos_m, off_m, bos_s, off_s = [], [], [], []
    for D in depths:
        R = load(f("softmax1", D))
        b = per_run(R, lambda r: np.mean([h["bos_mass"] for h in r["heads"]]))
        o = per_run(R, lambda r: np.mean([1 - h["total_mass"] for h in r["heads"]]))
        bos_m.append(b.mean()); bos_s.append(b.std()); off_m.append(o.mean()); off_s.append(o.std())
    axes[1].errorbar(depths, bos_m, yerr=bos_s, fmt="-o", color=COL["softmax1"], capsize=2,
                     label="mass on BOS")
    axes[1].errorbar(depths, off_m, yerr=off_s, fmt="--s", color=INK2, capsize=2,
                     label="mass on null slot")
    axes[1].set_xlabel("number of layers (4 heads each)")
    axes[1].set_ylabel("attention mass, non-trigger rows")
    axes[1].set_ylim(0, 0.8)
    axes[1].set_xticks(depths)
    axes[1].legend(loc="upper right")
    axes[1].set_title("(b) Softmax-off-by-one: BOS vs. null slot")
    fig.tight_layout()
    save(fig, out, "fig_depth.pdf")


# --------------------------------------------------------------------------- Fig 3
def fig_gateinit(out):
    biases = [-4, -2, 0, 2, 4]
    files = {-4: "results/phase2/G4L4Hbm4_gated.json", -2: "results/phase2/G4L4Hbm2_gated.json",
             0: "results/4L4H_gated.json", 2: "results/phase2/G4L4Hbp2_gated.json",
             4: "results/phase2/G4L4Hbp4_gated.json"}
    init = [1 / (1 + np.exp(-b)) for b in biases]
    sf, sfs, go, gt = [], [], [], []
    for b in biases:
        R = load(files[b])
        v = per_run(R, sink_frac)
        sf.append(v.mean()); sfs.append(v.std())
        go.append(np.mean([h["gate"] for r in R for h in r["heads"]]))
        gt.append(np.mean([h["gate_trig"] for r in R for h in r["heads"]]))
    x = np.arange(len(biases))
    labels = [f"{g:.2f}" for g in init]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.1))
    axes[0].bar(x, sf, 0.55, yerr=sfs, color=COL["gated"], capsize=2, edgecolor="white", linewidth=1)
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel("initial gate value  $\\sigma(b_0)$")
    axes[0].set_ylabel("fraction of sink heads")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("(a) Sinks vs. gate initialisation (4L4H)")
    axes[1].plot(x, init, ":", color=INK2, label="initial value")
    axes[1].plot(x, go, "-o", color=COL["gated"], label="final gate, non-trigger rows")
    axes[1].plot(x, gt, "--s", color=INK, label="final gate, trigger row")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("initial gate value  $\\sigma(b_0)$")
    axes[1].set_ylabel("gate value")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="lower right")
    axes[1].set_title("(b) The gate stays near its initialisation")
    fig.tight_layout()
    save(fig, out, "fig_gateinit.pdf")


# --------------------------------------------------------------------------- Fig 4
def fig_timelines(out):
    panels = [("(a) Softmax", "results/phase2/D4H4_softmax.json", "none"),
              ("(b) Gated, init 0.50", "results/phase2/G4L4Hbp0_gated.json", "gate"),
              ("(c) Gated, init 0.02", "results/phase2/G4L4Hbm4_gated.json", "gate"),
              ("(d) Softmax1, 8 layers", "results/phase2/D8H4_softmax1.json", "null")]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.4), sharex="col")
    for c, (title, f, kind) in enumerate(panels):
        r = next(r for r in load(f) if r["seed"] == 0)
        s = np.array([t["step"] for t in r["track"]])
        V = np.array([[h for layer in t["heads"] for h in layer] for t in r["track"]], dtype=object)
        bos = V[:, :, 0].astype(float)
        sink = bos[-1] > SINK
        ax = axes[0][c]
        for h in range(bos.shape[1]):
            ax.plot(s, bos[:, h], color=COL["softmax"] if sink[h] else "#b5b4ad",
                    lw=1.0 if sink[h] else 0.8, alpha=0.9, zorder=3 if sink[h] else 2)
        ax.set_title(f"{title}\n{sink.sum()}/{len(sink)} sink heads", color=INK)
        ax.set_ylim(-0.02, 1.02)
        ax2 = axes[1][c]
        if kind == "gate":
            g = V[:, :, 2].astype(float)
            for h in range(g.shape[1]):
                ax2.plot(s, g[:, h], color=COL["gated"], lw=0.9, alpha=0.8)
            ax2.text(0.97, 0.93, "gate (non-trigger rows)", ha="right", va="top",
                     transform=ax2.transAxes, fontsize=6.5, color=INK2)
        elif kind == "null":
            m = 1 - V[:, :, 1].astype(float)
            for h in range(m.shape[1]):
                ax2.plot(s, m[:, h], color=COL["softmax1"], lw=0.9, alpha=0.8)
            ax2.text(0.03, 0.93, "null-slot mass", ha="left", va="top",
                     transform=ax2.transAxes, fontsize=6.5, color=INK2)
        else:
            ax2.text(0.5, 0.5, "no 'off' mechanism", ha="center", va="center",
                     transform=ax2.transAxes, fontsize=7, color=INK2)
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_xlabel("training step")
        ax2.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    axes[0][0].set_ylabel("BOS mass")
    axes[1][0].set_ylabel("off-mechanism")
    fig.tight_layout()
    save(fig, out, "fig_timelines.pdf")


# --------------------------------------------------------------------------- Fig 5
def fig_probe(out):
    runs = json.load(open("results/phase2/probe_1L1H.json"))["runs"]
    T = min(len(r["trace"]) for r in runs) - 1
    steps = np.array([t["step"] for t in runs[0]["trace"][:T]])

    def stack(k, sub=None):
        return np.array([[(t[k][0][0]) for t in r["trace"][:T]] for r in runs])

    sp, gp, bos, gate = stack("sink_pull"), stack("gate_pull"), stack("bos"), stack("gate")
    keep = steps <= 700
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.1))
    for arr, lab, col, ls in [(sp, "sink pull", COL["softmax"], "-"),
                              (gp, "gate pull", COL["gated"], "--")]:
        m, s = arr.mean(0)[keep] * 1e4, arr.std(0)[keep] * 1e4
        axes[0].plot(steps[keep], m, ls, color=col, label=lab)
        axes[0].fill_between(steps[keep], m - s, m + s, color=col, alpha=0.15, lw=0)
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("gradient ($\\times 10^{-4}$)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("(a) Loss pulls harder on the gate early on")
    axes[1].plot(steps[keep], bos.mean(0)[keep], "-", color=COL["softmax"], label="BOS mass")
    axes[1].plot(steps[keep], gate.mean(0)[keep], "--", color=COL["gated"], label="gate (non-trigger)")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("value")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="lower right")
    axes[1].set_title("(b) ... yet the sink is what forms")
    fig.tight_layout()
    save(fig, out, "fig_probe.pdf")


# --------------------------------------------------------------------------- Fig 6
def fig_gatelr(out):
    mults = [1, 3, 10, 30, 100]

    def f(cfg, m):
        if cfg == "1L1H":
            return f"results/phase2/glr{m}_1L1H_gated.json"
        return "results/4L4H_gated.json" if m == 1 else f"results/phase2/glr{m}_4L4H_gated.json"

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    for cfg, ls, mk in [("1L1H", "--", "s"), ("4L4H", "-", "o")]:
        bos = [per_run(load(f(cfg, m)), lambda r: np.mean([h["bos_mass"] for h in r["heads"]])) for m in mults]
        stp = [per_run(load(f(cfg, m)), lambda r: r["steps"]) for m in mults]
        axes[0].errorbar(mults, [b.mean() for b in bos], yerr=[b.std() for b in bos], fmt=ls + mk,
                         color=COL["gated"] if cfg == "4L4H" else INK2, capsize=2, label=cfg)
        axes[1].errorbar(mults, [s.mean() for s in stp], yerr=[s.std() for s in stp], fmt=ls + mk,
                         color=COL["gated"] if cfg == "4L4H" else INK2, capsize=2, label=cfg)
    rows = [i for i in range(1, 16) if i != 7]           # uniform attention over keys 0..i
    uni = np.mean([1 / (i + 1) for i in rows])
    axes[0].axhline(uni, color=INK2, lw=0.8, ls=":", zorder=0)
    axes[0].text(1.05, uni - 0.09, "uniform attention", fontsize=6, color=INK2)
    axes[0].set_ylabel("mean BOS mass")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("(a) Sink strength")
    axes[1].set_ylabel("steps to converge")
    axes[1].set_yscale("log")
    axes[1].set_title("(b) Training speed")
    R4 = [load(f("4L4H", m)) for m in mults]
    axes[2].plot(mults, [np.mean([h["gate"] for r in R for h in r["heads"]]) for R in R4], "-o",
                 color=COL["gated"], label="non-trigger rows")
    axes[2].plot(mults, [np.mean([h["gate_trig"] for r in R for h in r["heads"]]) for R in R4], "--s",
                 color=INK, label="trigger row")
    axes[2].set_ylabel("final gate value (4L4H)")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("(c) The gate becomes a switch")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(mults, [f"×{m}" for m in mults])
        ax.minorticks_off()
        ax.set_xlabel("gate learning-rate multiplier")
        ax.legend(loc="best")
    fig.tight_layout()
    save(fig, out, "fig_gatelr.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for fn in [fig_repro, fig_depth, fig_gateinit, fig_timelines, fig_probe, fig_gatelr]:
        fn(a.out)


if __name__ == "__main__":
    main()
