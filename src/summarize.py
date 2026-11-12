"""Recompute every number quoted in the report from the raw results, and print it.

  python src/summarize.py            # prints all tables; writes results/summary.json

Used to verify the report: every table entry in report/main.tex is printed here.
"""
import glob
import json
import os

import numpy as np

SINK = 0.6


def load(f):
    return json.load(open(f))["results"]


def race(pattern):
    R = []
    for f in sorted(glob.glob(f"results/race/{pattern}.json")):
        R += json.load(open(f))["results"]
    return R


def summ(R):
    """Aggregate a list of runs of one configuration."""
    if not R:
        return None
    heads = [h for r in R for h in r["heads"]]
    out = dict(n=len(R), conv=sum(r["converged"] for r in R),
               steps=float(np.mean([r["steps"] for r in R])),
               steps_sd=float(np.std([r["steps"] for r in R])),
               sinks=sum(h["bos_mass"] > SINK for h in heads), heads=len(heads),
               mean_bos=float(np.mean([h["bos_mass"] for h in heads])),
               max_bos=float(np.mean([max(h["bos_mass"] for h in r["heads"]) for r in R])),
               test_linf=float(np.mean([r["test_linf"] for r in R])))
    if "gate" in heads[0]:
        out["gate"] = float(np.mean([h["gate"] for h in heads]))
        out["gate_trig"] = float(np.mean([h["gate_trig"] for h in heads]))
        if "gate_w" in heads[0]:
            out["gate_w"] = float(np.mean([h["gate_w"] for h in heads]))
            out["gate_trig_w"] = float(np.mean([h["gate_trig_w"] for h in heads]))
    if "sink_gap" in heads[0]:
        out["sink_gap"] = float(np.mean([h["sink_gap"] for h in heads]))
    if "gate_logit" in heads[0]:
        out["gate_logit"] = float(np.mean([h["gate_logit"] for h in heads]))
    return out


def fmt(name, s):
    if s is None:
        return f"{name:34s} (missing)"
    g = ""
    if "gate" in s:
        g = f" gate {s['gate']:.2f}/{s['gate_trig']:.2f}"
        if "gate_w" in s:
            g += f" (w {s['gate_w']:.2f}/{s['gate_trig_w']:.2f})"
    return (f"{name:34s} n={s['n']} conv={s['conv']} steps={s['steps']:7.0f}±{s['steps_sd']:5.0f} "
            f"sinks={s['sinks']:3d}/{s['heads']:3d} meanBOS={s['mean_bos']:.2f} maxBOS={s['max_bos']:.2f}{g}")


def main():
    S = {}
    print("== lambda* vs head width (1 layer, 4 heads, residual, lr 1e-4)")
    for dh in [4, 8, 16, 32, 64]:
        for lam in [1, 2, 4, 8, 16, 32, 64, 128]:
            s = summ(race(f"dh{dh}_lam{lam}_s*"))
            if s is None:
                continue
            S[f"dh{dh}_lam{lam}"] = s
            print(fmt(f"dh={dh} lam={lam}", s))
    print("== base learning rate (dh=16)")
    for lr in ["3e-5", "3e-4"]:
        for lam in [1, 2, 4, 8, 16, 32]:
            s = summ(race(f"lr{lr}_lam{lam}_s*"))
            S[f"lr{lr}_lam{lam}"] = s
            print(fmt(f"lr={lr} lam={lam}", s))
    print("== controls, 4 layers x 4 heads, dh=16")
    for tag in ["ctl_base_lam1", "ctl_lam10", "ctl_lam10_qk10", "ctl_lr1e-3", "ctl_scalar_lam1", "ctl_scalar_lam10"]:
        S[tag] = summ(race(f"{tag}_s*"))
        print(fmt(tag, S[tag]))
    print("== SGD (1 layer, 4 heads)")
    for tag in ["sgd_lr30_lam1", "sgd_lr30_softmax"]:
        S[tag] = summ(race(f"{tag}_s*"))
        print(fmt(tag, S[tag]))
    print("== gate parameterisations")
    for depth, pats in [("D1", {"linear1": "dh{dh}_lam1", "scaled": "fix_D1_scaled_dh{dh}",
                                "bilinear": "fix_D1_bilinear_dh{dh}"}),
                        ("L4", {"linear1": "fix_L4_linear_lam1_dh{dh}", "linear10": "fix_L4_linear_lam10_dh{dh}",
                                "scaled": "fix_L4_scaled_dh{dh}", "bilinear": "fix_L4_bilinear_dh{dh}"})]:
        for kind, pat in pats.items():
            for dh in [4, 16, 64]:
                p = pat.format(dh=dh)
                if depth == "L4" and dh == 16 and kind.startswith("linear"):
                    p = "ctl_base_lam1" if kind == "linear1" else "ctl_lam10"
                s = summ(race(p + "_s*"))
                S[f"fix_{depth}_{kind}_dh{dh}"] = s
                print(fmt(f"{depth} {kind} dh={dh}", s))
    for kind in ["scaled", "bilinear"]:
        S[f"fix_D8_{kind}_dh16"] = summ(race(f"fix_D8_{kind}_dh16_s*"))
        print(fmt(f"D8 {kind} dh=16", S[f"fix_D8_{kind}_dh16"]))
    S["fix_D8_linear1_dh16"] = summ([r for r in load("results/phase2/D8H4_gated.json") if r["seed"] < 3])
    print(fmt("D8 linear1 dh=16 (phase2, seeds 0-2)", S["fix_D8_linear1_dh16"]))
    print("== no BOS + learned positional embedding (Q/K)")
    for cfg in ["1L1H", "2L2H"]:
        R = race(f"pos_nobos_{cfg}_s*")
        for t in sorted({r["attn"] for r in R}):
            rs = [r for r in R if r["attn"] == t]
            s = summ(rs)
            s["max_key_mass"] = float(np.max([h["max_key_mass"] for r in rs for h in r["heads"]]))
            s["top_key"] = [h["top_key"] for r in rs for h in r["heads"]]
            S[f"pos_nobos_{cfg}_{t}"] = s
            print(fmt(f"noBOS+pos {cfg} {t}", s), f"maxkey={s['max_key_mass']:.2f} topkeys={s['top_key']}")
    # shared log-budget: sink gap + gate closing at the stopping point, per run (1-layer sweep)
    print("== shared log-budget (1 layer, 4 heads): per-head s* and -z* at the stopping point")
    pts = []
    for dh in [4, 8, 16, 32, 64]:
        for lam in [1, 2, 4, 8, 16, 32, 64, 128]:
            for r in race(f"dh{dh}_lam{lam}_s*"):
                for h in r["heads"]:
                    pts.append((dh, lam, h["sink_gap"], -h["gate_logit"], h["bos_mass"]))
    if pts:
        P = np.array(pts)
        tot = P[:, 2] + P[:, 3]
        print(f"   s*+(-z*): mean {tot.mean():.2f} sd {tot.std():.2f}; corr(s*,-z*) = {np.corrcoef(P[:, 2], P[:, 3])[0, 1]:.2f}")
        S["budget"] = dict(mean=float(tot.mean()), sd=float(tot.std()),
                           corr=float(np.corrcoef(P[:, 2], P[:, 3])[0, 1]))
    print("== language model (TinyStories); per variant, mean over seeds, final checkpoint")
    for prefix in ["lm", "lm2"]:
        for tag in ["softmax", "softmax1", "gated_lam1", "gated_lam10", "gated_scaled"]:
            E = [json.load(open(f)) for f in sorted(glob.glob(f"results/lm/{prefix}_{tag}_s*.json"))]
            if not E:
                continue
            last = [d["log"][-1] for d in E]
            ratio = [np.mean([a["max_pos0"] / a["max_rest"] for a in e["acts"][1:-1]]) for e in last]
            gate = [np.mean([h["gate"] for h in e["heads"]]) for e in last if "gate" in e["heads"][0]]
            r = dict(n=len(E), step=last[0]["step"], val_loss=float(np.mean([e["val_loss"] for e in last])),
                     val_loss_sd=float(np.std([e["val_loss"] for e in last])),
                     mean_bos=float(np.mean([e["mean_bos"] for e in last])),
                     max_head_bos=float(np.mean([max(h["bos"] for h in e["heads"]) for e in last])),
                     sink_frac=float(np.mean([e["sink_frac"] for e in last])),
                     anchor_frac=float(np.mean([e.get("anchor_frac", np.nan) for e in last])),
                     max_head_anykey=float(np.mean([max(h.get("maxkey", np.nan) for h in e["heads"]) for e in last])),
                     bos_act_ratio=float(np.mean(ratio)), massive_ratio=float(np.mean([e["massive_ratio"] for e in last])),
                     gate=float(np.mean(gate)) if gate else None, nparam=E[0]["nparam"])
            S[f"{prefix}_{tag}"] = r
            print(f"{prefix:4s} {tag:13s} n={r['n']} step {r['step']} val {r['val_loss']:.4f}+-{r['val_loss_sd']:.4f} "
                  f"meanBOS {r['mean_bos']:.3f} maxheadBOS {r['max_head_bos']:.3f} sinkfrac {r['sink_frac']:.2f} "
                  f"anykey {r['max_head_anykey']:.2f} BOSact/rest(L2-5) {r['bos_act_ratio']:.2f} massive {r['massive_ratio']:.0f}"
                  + (f" gate {r['gate']:.2f}" if gate else ""))
    json.dump(S, open("results/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
