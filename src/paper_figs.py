"""All figures used in the report (vector PDF), from results/ JSON files.

  python src/paper_figs.py --out report/figures
"""
import argparse
import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# fixed categorical order (validated reference palette, slots 1-3 + violet for relu)
COL = {"softmax": "#2a78d6", "softmax1": "#eb6834", "gated": "#1baf7a", "relu": "#6250d6"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SINK = 0.6  # a head is a "sink head" if mean BOS mass on non-trigger rows > 0.6
DH_COL = {4: "#86b6ef", 8: "#5598e7", 16: "#2a78d6", 32: "#1c5cab", 64: "#104281"}  # ordinal blue ramp
KIND_COL = {"linear1": "#8a8984", "linear10": COL["softmax1"], "scaled": COL["gated"], "bilinear": COL["relu"]}
KIND_LAB = {"linear1": "linear gate, $\\lambda$=1", "linear10": "linear gate, $\\lambda$=10",
            "scaled": "race-matched gate", "bilinear": "bilinear gate"}
sys.path.insert(0, os.path.dirname(__file__))

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
    runs10 = json.load(open("results/phase2/probe_1L1H_lam10.json"))["runs"]

    def stack(R, k):
        T = min(len(r["trace"]) for r in R) - 1
        st = np.array([t["step"] for t in R[0]["trace"][:T]])
        return st, np.array([[t[k][0][0] for t in r["trace"][:T]] for r in R])

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    st, sp = stack(runs, "sink_pull")
    _, gp = stack(runs, "gate_pull")
    _, gs = stack(runs, "gate_pull_sum")
    keep = st <= 700
    for arr, lab, col, ls in [(sp, "sink pull", COL["softmax"], "-"),
                              (gp, "gate pull, per coordinate", COL["gated"], "-"),
                              (gs, "gate pull, summed over 16", COL["gated"], ":")]:
        m, sd = arr.mean(0)[keep] * 1e4, arr.std(0)[keep] * 1e4
        axes[0].plot(st[keep], m, ls, color=col, label=lab)
        axes[0].fill_between(st[keep], m - sd, m + sd, color=col, alpha=0.12, lw=0)
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("gradient ($\\times 10^{-4}$)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("(a) Which way the loss pulls")
    # (b),(c): realised trajectories of the two routes vs the race theory (sink ~ t^2, gate ~ t)
    for ax, R, title in [(axes[1], runs, "(b) $\\lambda$=1: the sink wins"),
                         (axes[2], runs10, "(c) $\\lambda$=10: both routes share the work")]:
        st, gap = stack(R, "sink_gap")
        _, gl = stack(R, "gate_logit")
        ax.plot(st, gap.mean(0), "-", color=COL["softmax"], label="sink logit gap $s$")
        ax.plot(st, -gl.mean(0), "-", color=COL["gated"], label="gate closing $-z$")
        # early-phase fits (until the sink logit reaches 3, before the error saturates): s = a t^2, -z = b t
        h = gap.mean(0) <= 3.0
        h[: np.argmax(~h) if (~h).any() else len(h)] = True
        a = np.polyfit(st[h] ** 2, gap.mean(0)[h], 1)
        b = np.polyfit(st[h], -gl.mean(0)[h], 1)
        sf = st[st <= 1.4 * st[h].max()]
        ax.plot(sf, np.polyval(a, sf ** 2), ":", color=INK, lw=1.1, label="early fit $at^2$, $bt$")
        ax.plot(sf, np.polyval(b, sf), ":", color=INK, lw=1.1)
        ax.set_xlabel("training step")
        ax.set_title(title)
        ax.set_ylim(bottom=-0.5)
    axes[1].set_ylabel("logit")
    axes[1].legend(loc="upper left")
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


# --------------------------------------------------------------------------- race (Sec. 8)
def race(pattern):
    R = []
    for f in sorted(glob.glob(f"results/race/{pattern}.json")):
        R += json.load(open(f))["results"]
    return R


def mean_bos(r):
    return np.mean([h["bos_mass"] for h in r["heads"]])


def cross(xs, ys, level=SINK):
    """log-interpolated x where a decreasing curve ys(xs) first drops below level."""
    pts = [(x, y) for x, y in zip(xs, ys) if not np.isnan(y)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y0 >= level > y1:
            t = (y0 - level) / (y0 - y1)
            return math.exp(math.log(x0) + t * (math.log(x1) - math.log(x0)))
    return np.nan


def fig_race(out):
    from race_theory import lam_star, sink_share
    sim = json.load(open("results/theory/race_sim.json"))
    lams, dhs = [1, 2, 4, 8, 16, 32, 64, 128], [4, 8, 16, 32, 64]
    lf = np.geomspace(1, 128, 200)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    for dh in [4, 16, 64]:
        axes[0].plot(lf, [sink_share(dh, l)[1] for l in lf], "-", color=DH_COL[dh], label=f"$d_h$={dh}")
        S = [r for r in sim["sim"] if r["dh"] == dh and r["opt"] == "adam" and r["eta"] == 1e-3 and r["gate"] == "linear"]
        L = sorted({r["lam"] for r in S})
        axes[0].plot(L, [np.mean([r["bos"] for r in S if r["lam"] == l]) for l in L], "o", color=DH_COL[dh],
                     mfc="white", mew=1.2)
    axes[0].set_title("(a) Theory (line), Adam sim (o)")
    axes[0].set_ylabel("BOS mass when training stops")
    real_ls = {}
    for dh in dhs:
        m, sd = [], []
        for l in lams:
            v = [mean_bos(r) for r in race(f"dh{dh}_lam{l}_s*")]
            m.append(np.mean(v) if v else np.nan); sd.append(np.std(v) if v else np.nan)
        m, sd = np.array(m), np.array(sd)
        real_ls[dh] = m
        ok = ~np.isnan(m)
        axes[1].errorbar(np.array(lams)[ok], m[ok], yerr=sd[ok], fmt="-o", color=DH_COL[dh], capsize=2,
                         label=f"$d_h$={dh}")
    axes[1].set_title("(b) Trained, 1 layer $\\times$ 4 heads")
    for ax in axes[:2]:
        ax.axhline(SINK, color=INK2, lw=0.8, ls="--", zorder=0)
        ax.set_xscale("log"); ax.set_xticks(lams, [str(l) for l in lams]); ax.minorticks_off()
        ax.set_xlabel("gate learning-rate multiplier $\\lambda$")
        ax.set_ylim(0, 1.05)
    axes[1].legend(loc="lower left", ncol=1)
    # (c) critical lambda vs head width
    d = np.array(dhs, float)
    real = [cross(lams, list(real_ls[dh])) for dh in dhs]
    simc = []
    for dh in dhs:
        S = [r for r in sim["sim"] if r["dh"] == dh and r["opt"] == "adam" and r["eta"] == 1e-3 and r["gate"] == "linear"]
        L = sorted({r["lam"] for r in S})
        simc.append(cross(L, [np.mean([r["bos"] for r in S if r["lam"] == l]) for l in L]))
    real = np.array(real)
    fit = d <= 32
    c = float(np.exp(np.mean(np.log(real[fit] / np.array([lam_star(x) for x in d[fit]])))))
    axes[2].plot(d, [lam_star(x) for x in d], "-", color=INK2, label="closed form $\\propto\\sqrt{d_h}$")
    axes[2].plot(d, [c * lam_star(x) for x in d], "--", color=COL["softmax"], lw=1.0,
                 label=f"closed form $\\times${c:.1f}")
    axes[2].plot(d, simc, "o", color=INK2, mfc="white", mew=1.2, label="Adam sim")
    axes[2].plot(d[fit], real[fit], "s", color=COL["softmax"], ms=5, label="trained model")
    axes[2].plot(d[~fit], real[~fit], "s", color=COL["softmax"], mfc="white", ms=5, mew=1.2)
    k = np.polyfit(np.log(d[fit]), np.log(real[fit]), 1)[0]
    axes[2].text(0.04, 0.95, f"fitted slope {k:.2f} ($d_h\\leq$32; theory 0.5)", transform=axes[2].transAxes,
                 fontsize=6.3, color=INK2, va="top")
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].set_xticks(dhs, [str(x) for x in dhs]); axes[2].minorticks_off()
    axes[2].set_xlabel("head width $d_h$")
    axes[2].set_ylabel("critical $\\lambda^*$")
    axes[2].set_title("(c) $\\lambda^*$ grows like $\\sqrt{d_h}$")
    axes[2].set_yticks([3, 10, 30], ["3", "10", "30"])
    axes[2].set_ylim(0.45, 110)
    axes[2].legend(loc="lower right", fontsize=5.6)
    fig.tight_layout()
    save(fig, out, "fig_race.pdf")
    return dict(real=dict(zip(dhs, real.tolist())), sim=dict(zip(dhs, simc)), slope=k, offset=c)


def fig_lr(out):
    lams = [1, 2, 4, 8, 16, 32]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.2))
    lr_col = {"3e-5": "#86b6ef", "1e-4": "#2a78d6", "3e-4": "#104281"}
    for lr, col in lr_col.items():
        pat = (lambda l: f"dh16_lam{l}_s*") if lr == "1e-4" else (lambda l, lr=lr: f"lr{lr}_lam{l}_s*")
        v = [[mean_bos(r) for r in race(pat(l))] for l in lams]
        axes[0].errorbar(lams, [np.mean(x) for x in v], yerr=[np.std(x) for x in v], fmt="-o", color=col,
                         capsize=2, label=f"base lr {lr}")
    axes[0].axhline(SINK, color=INK2, lw=0.8, ls="--", zorder=0)
    axes[0].set_xscale("log"); axes[0].set_xticks(lams, [str(l) for l in lams]); axes[0].minorticks_off()
    axes[0].set_xlabel("gate lr multiplier $\\lambda$"); axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("BOS mass"); axes[0].legend(loc="lower left")
    axes[0].set_title("(a) The base learning rate cancels")
    # (b) shared log-budget: per head, sink logit gap vs. gate closing at the stopping point
    tot = []
    for dh in [4, 8, 16, 32, 64]:
        xs, ys = [], []
        for l in [1, 2, 4, 8, 16, 32, 64, 128]:
            for r in race(f"dh{dh}_lam{l}_s*"):
                for h in r["heads"]:
                    xs.append(-h["gate_logit"]); ys.append(h["sink_gap"])
        tot += [x + y for x, y in zip(xs, ys)]
        axes[1].plot(xs, ys, "o", ms=2.6, color=DH_COL[dh], alpha=0.8, label=f"$d_h$={dh}", mew=0)
    c = float(np.mean(tot))
    xx = np.linspace(0, c, 50)
    axes[1].plot(xx, c - xx, "--", color=INK2, lw=0.9, label=f"$s+(-z)={c:.1f}$")
    axes[1].set_xlabel("gate closing $-z^*$"); axes[1].set_ylabel("sink logit gap $s^*$")
    axes[1].set_xlim(left=-0.3); axes[1].set_ylim(bottom=-0.3)
    axes[1].legend(loc="upper right", ncol=2, fontsize=5.8)
    axes[1].set_title("(b) One shared log-budget")
    fig.tight_layout()
    save(fig, out, "fig_lr.pdf")


def fix_runs(kind, dh, depth="L4"):
    if depth == "L4":
        if kind in ("linear1", "linear10"):
            lam = 1 if kind == "linear1" else 10
            if dh == 16:
                return race("ctl_base_lam1_s*" if lam == 1 else "ctl_lam10_s*")
            return race(f"fix_L4_linear_lam{lam}_dh{dh}_s*")
        return race(f"fix_L4_{kind}_dh{dh}_s*")
    if depth == "D1":
        if kind == "linear1":
            return race(f"dh{dh}_lam1_s*")
        return race(f"fix_D1_{kind}_dh{dh}_s*")
    if depth == "D8":
        if kind == "linear1":
            return [r for r in load("results/phase2/D8H4_gated.json") if r["seed"] < 3]
        return race(f"fix_D8_{kind}_dh16_s*")


def fig_fix(out):
    dhs, kinds = [4, 16, 64], ["linear1", "linear10", "scaled", "bilinear"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    x = np.arange(len(dhs))
    w = 0.2
    for k, kind in enumerate(kinds):
        sf = [per_run(fix_runs(kind, dh), sink_frac) for dh in dhs]
        st = [per_run(fix_runs(kind, dh), lambda r: r["steps"]) for dh in dhs]
        off = (k - 1.5) * w
        axes[0].bar(x + off, [v.mean() for v in sf], w * 0.9, yerr=[v.std() for v in sf], color=KIND_COL[kind],
                    capsize=1.5, label=KIND_LAB[kind], edgecolor="white", linewidth=0.5)
        axes[1].bar(x + off, [v.mean() for v in st], w * 0.9, yerr=[v.std() for v in st], color=KIND_COL[kind],
                    capsize=1.5, edgecolor="white", linewidth=0.5)
        for xi, v in zip(x + off, sf):
            if v.mean() == 0:
                axes[0].text(xi, 0.015, "0", ha="center", va="bottom", fontsize=6, color=INK2)
    for ax in axes[:2]:
        ax.set_xticks(x, [f"$d_h$={d}" for d in dhs])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("fraction of sink heads"); axes[0].set_ylim(0, 1.05)
    axes[0].set_title("(a) Sinks, 4 layers $\\times$ 4 heads")
    axes[1].set_ylabel("steps to converge"); axes[1].set_yscale("log")
    axes[1].set_title("(b) Training speed")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    # (c) depth, d_h = 16
    depths = [("D1", 1), ("L4", 4), ("D8", 8)]
    for kind in ["linear1", "scaled", "bilinear"]:
        v = [per_run(fix_runs(kind, 16, dp), mean_bos) for dp, _ in depths]
        axes[2].errorbar([d for _, d in depths], [a.mean() for a in v], yerr=[a.std() for a in v], fmt="-o",
                         color=KIND_COL[kind], capsize=2)
    axes[2].axhline(SINK, color=INK2, lw=0.8, ls="--", zorder=0)
    axes[2].set_xticks([1, 4, 8]); axes[2].set_xlabel("layers ($d_h$=16, 4 heads)")
    axes[2].set_ylabel("mean BOS mass"); axes[2].set_ylim(0, 1.05)
    axes[2].set_title("(c) Across depth")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out, "fig_fix.pdf")


LM_VARIANTS = [("softmax", "softmax", COL["softmax"]), ("softmax1", "off-by-one", "#8a8984"),
               ("gated_lam1", "linear gate, $\\lambda$=1", COL["softmax1"]),
               ("gated_lam10", "linear gate, $\\lambda$=10", COL["relu"]),
               ("gated_scaled", "race-matched gate", COL["gated"])]


def lm_logs(prefix, tag):
    return [json.load(open(f))["log"] for f in sorted(glob.glob(f"results/lm/{prefix}_{tag}_s*.json"))]


def fig_lm(out, prefix="lm", name="fig_lm.pdf"):
    if not glob.glob(f"results/lm/{prefix}_*.json"):
        print(f"skip {name} (no results/lm/{prefix}_*)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    for tag, lab, col in LM_VARIANTS:
        L = lm_logs(prefix, tag)
        if not L:
            continue
        T = min(len(l) for l in L)
        st = np.array([e["step"] for e in L[0][:T]])
        # (a) BOS-specific massive activation, final checkpoint, per layer
        rat = np.array([[a["max_pos0"] / a["max_rest"] for a in l[T - 1]["acts"]] for l in L])
        axes[0].plot(np.arange(1, rat.shape[1] + 1), rat.mean(0), "-o", color=col, label=lab)
        # (b) strongest BOS head over training
        mb = np.array([[max(h["bos"] for h in e["heads"]) for e in l[:T]] for l in L])
        axes[1].plot(st, mb.mean(0), "-", color=col)
        # (c) validation loss
        v = np.array([[e["val_loss"] for e in l[:T]] for l in L])
        keep = st >= st[-1] * 0.25
        axes[2].plot(st[keep], v.mean(0)[keep], "-", color=col)
    axes[0].axhline(1.0, color=INK2, lw=0.8, ls="--", zorder=0)
    axes[0].set_xlabel("layer"); axes[0].set_ylabel("max $|h|$ at BOS / elsewhere")
    axes[0].set_title("(a) BOS massive activation")
    axes[1].set_ylabel("strongest head's BOS attention"); axes[1].set_ylim(bottom=0)
    axes[1].set_title("(b) Sink strength")
    axes[2].set_ylabel("validation loss (nats/char)")
    axes[2].set_title("(c) Language modelling")
    for ax in axes[1:]:
        ax.set_xlabel("training step")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out, name)


def fig_lm2(out):
    fig_lm(out, prefix="lm2", name="fig_lm2.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--only", nargs="*", default=None, help="subset of figure names, e.g. fig_race")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fns = [fig_repro, fig_depth, fig_gateinit, fig_timelines, fig_probe, fig_gatelr, fig_race, fig_lr, fig_fix, fig_lm, fig_lm2]
    for fn in fns:
        if a.only is None or fn.__name__ in a.only:
            fn(a.out)


if __name__ == "__main__":
    main()
