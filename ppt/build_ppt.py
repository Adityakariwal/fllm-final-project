"""Build the presentation from the course template (6 slides, same order and layouts).

  python ppt/build_ppt.py          # -> ppt/G28_presentation.pptx

Needs: python-pptx, pdftoppm (poppler) to rasterise the report figures, results/summary.json
(python src/summarize.py).
"""
import json
import os
import subprocess

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "ppt", "Project_Presentation_Template.pptx")
OUT = os.path.join(ROOT, "ppt", "G28_presentation.pptx")
ASSETS = os.path.join(ROOT, "ppt", "assets")
INK = RGBColor(0x1F, 0x1F, 0x1F)
MUTED = RGBColor(0x52, 0x51, 0x4E)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
GREEN = RGBColor(0x1B, 0xAF, 0x7A)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
CODE_BG = RGBColor(0xF4, 0xF4, 0xF2)

S = json.load(open(os.path.join(ROOT, "results", "summary.json")))


def png(name, dpi=220):
    os.makedirs(ASSETS, exist_ok=True)
    out = os.path.join(ASSETS, name)
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile",
                    os.path.join(ROOT, "report", "figures", name + ".pdf"), out], check=True)
    return out + ".png"


def body(slide):
    """The template's free text box below the title (not a numbered placeholder)."""
    return [sh for sh in slide.shapes if sh.has_text_frame and sh.left < Inches(1.5) and sh.top > Inches(1)][0]


def fill(tf, items, size=14, space=4):
    """items: list of (level, text) or (level, [(text, bold), ...])."""
    tf.clear()
    tf.word_wrap = True
    first = True
    for level, content in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        p.space_after = Pt(space)
        runs = [(content, False)] if isinstance(content, str) else content
        for text, bold in runs:
            r = p.add_run()
            r.text = text
            r.font.size = Pt(size - 2 * level)
            r.font.bold = bold
            r.font.color.rgb = INK if level == 0 else MUTED


def box(slide, x, y, w, h, text, fill_rgb, size=12, bold=False, color=RGBColor(0xFF, 0xFF, 0xFF)):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill_rgb
    sh.line.fill.background()
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold and i == 0
        r.font.color.rgb = color
    return sh


def arrow(slide, x0, y0, x1, y1):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x0), Inches(y0), Inches(x1), Inches(y1))
    c.line.color.rgb = MUTED
    c.line.width = Pt(1.5)
    ln = c.line._get_or_add_ln()
    from pptx.oxml.ns import qn
    tail = ln.find(qn("a:tailEnd"))
    if tail is None:
        tail = ln.makeelement(qn("a:tailEnd"), {})
        ln.append(tail)
    tail.set("type", "triangle")
    return c


def code(slide, x, y, w, h, title, src, size=10):
    t = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.3))
    r = t.text_frame.paragraphs[0].add_run()
    r.text = title
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.color.rgb = INK
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y + 0.32), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = CODE_BG
    sh.line.color.rgb = RGBColor(0xD9, 0xD8, 0xD4)
    tf = sh.text_frame
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_right = Inches(0.1)
    tf.margin_top = Inches(0.06)
    for i, line in enumerate(src.strip("\n").split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        r = p.add_run()
        r.text = line if line else " "
        r.font.size = Pt(size)
        r.font.name = "Consolas"
        r.font.color.rgb = INK


def lm_rows():
    rows = []
    for tag, lab in [("softmax", "softmax"), ("softmax1", "off-by-one"), ("gated_lam1", "gate, λ=1"),
                     ("gated_lam10", "gate, λ=10"), ("gated_scaled", "race-matched gate")]:
        v = [S[k] for k in S if k.startswith(f"lm_{tag}_s")]
        if v:
            avg = lambda key: sum(x[key] for x in v) / len(v)
            rows.append((lab, avg("val_loss"), avg("mean_bos"), avg("sink_frac"), avg("massive_ratio")))
    return rows


def main():
    prs = Presentation(TEMPLATE)
    s1, s2, s3, s4, s5, s6 = prs.slides

    # ---------------------------------------------------------------- 1. title
    title, sub = [sh for sh in s1.shapes if sh.is_placeholder][:2]
    title.text_frame.text = "Racing to the Sink: Why Gated Attention Still Forms Attention Sinks"
    for r in title.text_frame.paragraphs[0].runs:
        r.font.size = Pt(30)
    fill(sub.text_frame, [(0, [("Group Number: ", True), ("G28", False)]),
                          (0, [("Names: ", True), ("Aditya Kariwal, Manish Kundu, Samyak Sheersh", False)]),
                          (0, [("Roll Numbers: ", True), (ROLLS, False)]),
                          (0, [("Base paper: ", True), ("Ran-Milo (2026), Attention Sinks Are Provably Necessary "
                                                         "in Softmax Transformers, arXiv:2603.11487", False)])],
         size=16)

    # ---------------------------------------------------------------- 2. problem
    fill(body(s2).text_frame, [
        (0, [("Task (base paper). ", True), ("Sequence x ∈ ℝ^{L×n}, L = n = 16. Position 0 = BOS (e_bos), one "
             "trigger at random j (e_trig), all other tokens e_cont + 13 i.i.d. U(−1,1) content features.", False)]),
        (0, [("Target. ", True), ("y⁽ʲ⁾ = (1/j) Σ_{k=1..j} x⁽ᵏ⁾ at the trigger, y⁽ⁱ⁾ = 0 everywhere else (a trigger-"
             "conditional no-op). Loss: ℓ∞ error; training stops when it is below 5·10⁻³.", False)]),
        (0, [("Theorem (Ran-Milo). ", True), ("Every low-loss softmax model puts attention α_{i,BOS} → 1 on every "
             "non-trigger row: the attention sink is necessary. ReLU attention needs no sink. The paper's advice: "
             "remove sinks with explicit gating or un-normalised attention.", False)]),
        (0, [("Our question. ", True), ("A gated head can output zero without a sink (we prove it). Does training "
             "actually use the gate? If not, what decides which way a head switches off, and can we control it?",
             False)]),
        (0, [("Input → process → output. ", True), ("Synthetic sequences → attention-only transformer (D layers × H "
             "heads) → per-position vectors; we measure where each head's attention goes.", False)]),
        (0, [("Data pipeline. ", True), ("Synthetic task: 128 fresh sequences per step, generated on the fly. "
             "Language model: TinyStories, character level, each sample = [BOS] + the first 255 characters of a story.",
             False)]),
    ], size=16, space=8)

    # ---------------------------------------------------------------- 3. architecture
    b3 = body(s3)
    b3.width = Inches(6.3)
    fill(b3.text_frame, [
        (0, [("Model (as in the base paper). ", True), ("Attention-only transformer, D layers × H heads of width d_h, "
             "residual stream for D > 1, no MLP / LayerNorm / positional encoding.", False)]),
        (0, [("Head output. ", True), ("W_O ( g_i ⊙ Σ_k α_ik v_k ), with α from one of:", False)]),
        (1, "softmax (base), ReLU (base, Thm 3), softmax-off-by-one, sigmoid"),
        (1, "gated softmax: g_i = σ(W_G x_i + b), elementwise per head (Qiu et al. 2025)"),
        (0, [("Gate parameterisations we study. ", True), ("", False)]),
        (1, "linear gate + learning-rate multiplier λ on (W_G, b) only"),
        (1, "race-matched gate: σ(κ(W_G x + b)), κ = 2.5·√d_h, init/κ"),
        (1, "bilinear gate: z = A_h(B_h x) + b (product of two factors, like the sink)"),
        (0, [("Language model. ", True), ("Char-level GPT: 6 layers, d = 384, 6 heads × 64, pre-LN, MLP, "
             "context 256, AdamW, 8k steps; same attention variants.", False)]),
    ], size=14, space=6)
    # diagram: two routes to "off"
    X = 7.55
    box(s3, X + 1.35, 1.55, 2.6, 0.75, "non-trigger row\nmust output 0", MUTED, size=12, bold=True)
    box(s3, X, 2.85, 2.45, 1.25, "Sink route\ns = q·k_BOS = Σ_r Q_r K_r\nbilinear: s ≈ d_h (ηt)²", BLUE, size=11,
        bold=True)
    box(s3, X + 2.85, 2.85, 2.45, 1.25, "Gate route\nz = W_G x + b\nlinear: −z ≈ 2λ ηt", GREEN, size=11, bold=True)
    box(s3, X + 0.6, 4.6, 4.1, 0.95, "Shared log-budget at convergence\ns* + (−z*) ≈ log(C·i / ε)", ORANGE,
        size=11, bold=True)
    arrow(s3, X + 2.2, 2.3, X + 1.2, 2.85)
    arrow(s3, X + 3.1, 2.3, X + 4.1, 2.85)
    arrow(s3, X + 1.2, 4.1, X + 2.0, 4.6)
    arrow(s3, X + 4.1, 4.1, X + 3.3, 4.6)
    t = s3.shapes.add_textbox(Inches(X - 0.1), Inches(5.65), Inches(5.6), Inches(0.9))
    fill(t.text_frame, [(0, "The faster route takes the larger share. A head is a sink if s* > log(1.5·i), "
                           "which gives λ* ∝ √d_h.")], size=12)

    # ---------------------------------------------------------------- 4. contribution code
    b4 = body(s4)
    b4.height = Inches(0.55)
    fill(b4.text_frame, [(0, "Our implementation (PyTorch, written from scratch). Excerpts from src/sinks.py, "
                             "src/race_theory.py and src/probe.py.")], size=13)
    code(s4, 0.93, 1.95, 6.15, 2.6, "Gate parameterisations (src/sinks.py)", '''
if gate_kind == "bilinear":                       # z = A_h (B_h x) + b
    self.W_GB = nn.Linear(d, H * dh, bias=False)
    self.W_GA = nn.Parameter(torch.empty(H, dh, dh))
else:
    self.W_G = nn.Linear(d, H if scalar else H * dh, bias=True)
    if gate_kind == "scaled":                     # race-matched gate
        self.kappa = gate_scale * math.sqrt(dh)
...
self.W_G.weight.data.div_(self.kappa)             # muP: init/kappa, x kappa
def gate_logits(self, x):
    return self.kappa * self.W_G(x)  # == gate lr x kappa under Adam
''', size=9)
    code(s4, 7.25, 1.95, 5.6, 2.6, "Racing the routes (src/sinks.py: train)", '''
gate_p = [p for n, p in named if "W_G" in n]
qk_p   = [p for n, p in named if "W_Q" in n or "W_K" in n]
groups = [dict(params=other_p, lr=lr),
          dict(params=qk_p,   lr=lr * qk_lr_mult),   # speed up sink
          dict(params=gate_p, lr=lr * gate_lr_mult)] # speed up gate
opt = (torch.optim.Adam(groups, betas=(0.9, 0.95))
       if opt == "adam" else torch.optim.SGD(groups, momentum=0.9))
# stop when the l_inf error on a batch < 5e-3 (paper's rule)
''', size=9)
    code(s4, 0.93, 4.95, 6.15, 1.95, "Closed-form critical multiplier (src/race_theory.py)", '''
def lam_star(dh, i=8, eps=5e-3, vbar=1.0, a0=0.6):
    T  = math.log(vbar * i / eps)        # shared log-budget T'
    S0 = math.log(i * a0 / (1 - a0))     # sink logit of a "sink head"
    return (T - S0) * math.sqrt(dh) / (2 * math.sqrt(S0))
''', size=9)
    code(s4, 7.25, 4.95, 5.6, 1.95, "Zero-valued gradient probes (src/probe.py)", '''
p_s = torch.zeros(H, requires_grad=True)   # added to BOS logit
p_g = torch.zeros(H, requires_grad=True)   # added to gate logit
layer.probe = (row_mask, p_s, p_g)         # non-trigger rows only
gs = torch.autograd.grad(loss, ps + pg)
gate_pull = g / layer.dh                   # per coordinate (fix)
''', size=9)

    # ---------------------------------------------------------------- 5. benchmarks & evaluation
    b5 = body(s5)
    b5.width = Inches(6.0)
    fill(b5.text_frame, [
        (0, [("Benchmark methods", True)]),
        (1, "Base paper: softmax attention (sink necessary) and ReLU attention (sink-free, Thm 3)"),
        (1, "Off-switch baselines: softmax-off-by-one (Miller 2023), gated attention (Qiu et al. 2025), sigmoid"),
        (1, "Ours: gate learning-rate multiplier λ, race-matched gate, bilinear gate; controls with Q/K-lr ×10, "
            "all-lr ×10, SGD, scalar gate"),
        (0, [("Evaluation metrics", True)]),
        (1, "BOS attention mass on non-trigger rows; sink head = mass > 0.6"),
        (1, "Gate value on non-trigger / trigger rows (importance-weighted over coordinates)"),
        (1, "Steps to reach ℓ∞ < 5·10⁻³; sink logit s* and gate logit z* when training stops"),
        (1, "Critical multiplier λ*: where mean BOS mass crosses 0.6"),
        (1, "LM: validation loss (nats/char), BOS attention, massive-activation ratio max|h| / median|h|"),
        (0, [("Protocol", True)]),
        (1, "3–5 seeds per cell; ~550 trained models; every number recomputed by src/summarize.py"),
    ], size=14, space=4)
    c = S["ctl_base_lam1"], S["ctl_lam10"], S["ctl_lam10_qk10"], S["ctl_lr1e-3"]
    rows = [("Configuration (4 layers × 4 heads)", "Sink heads", "Steps"),
            ("baseline, λ = 1", f"{c[0]['sinks']}/{c[0]['heads']}", f"{c[0]['steps'] / 1e3:.1f}k"),
            ("gate lr ×10 (λ = 10)", f"{c[1]['sinks']}/{c[1]['heads']}", f"{c[1]['steps'] / 1e3:.1f}k"),
            ("gate ×10 and Q/K ×10", f"{c[2]['sinks']}/{c[2]['heads']}", f"{c[2]['steps'] / 1e3:.1f}k"),
            ("all learning rates ×10", f"{c[3]['sinks']}/{c[3]['heads']}", f"{c[3]['steps'] / 1e3:.1f}k")]
    tb = s5.shapes.add_table(len(rows), 3, Inches(7.2), Inches(1.6), Inches(5.6), Inches(2.2)).table
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            cell = tb.cell(i, j)
            cell.text = v
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(13)
                    r.font.bold = i == 0
    tb.columns[0].width = Inches(3.3)
    tb.columns[1].width = Inches(1.15)
    tb.columns[2].width = Inches(1.15)
    t = s5.shapes.add_textbox(Inches(7.2), Inches(3.95), Inches(5.6), Inches(2.6))
    fill(t.text_frame, [(0, [("Controls make the race causal. ", True),
                             ("Speeding up only the gate removes every sink; speeding up the sink's factors as well "
                              "brings every sink back. A larger learning rate overall trains as fast but keeps "
                              "the sinks.", False)])], size=13)

    # ---------------------------------------------------------------- 6. results & discussion
    b6 = body(s6)
    b6.width = Inches(5.7)
    lm = lm_rows()
    lm_items = []
    if lm:
        soft, g1 = lm[0], next((r for r in lm if r[0] == "gate, λ=1"), None)
        lm_items = [(0, [("Language model: ", True), (LM_SENTENCE, False)])]
    fill(b6.text_frame, [
        (0, [("Sinks are preferred, not just necessary: ", True),
             ("gated attention keeps BOS sinks at depths 1–8 and every gate init, although we prove a sink-free "
              "zero-loss solution exists.", False)]),
        (0, [("A quantitative race: ", True),
             (f"shared budget s*+(−z*) = {S['budget']['mean']:.1f} ± {S['budget']['sd']:.1f} (r = "
              f"{S['budget']['corr']:.2f}); outcome independent of the base lr; λ* ∝ √d_h (slope 0.55 vs 0.5), "
              "≈5× the closed-form constant.", False)]),
        (0, [("What fails: ", True), ("SGD does not remove sinks; at 4 layers λ* is ~width-independent, so the "
                                       "√d_h-scaled gate fails at d_h = 4.", False)]),
        *lm_items,
        (0, [("Limitations: ", True), ("synthetic task and attention-only models; first-order theory "
                                        "(constant off by ~5×); small LM.", False)]),
        (0, [("Conclusion: ", True), ("an off switch makes sinks unnecessary but not absent: the gate must also "
                                       "learn fast enough. A gate-lr multiplier of ~10 (deeper) to ~30–50 (1 layer) "
                                       "decides the race.", False)]),
    ], size=13, space=6)
    s6.shapes.add_picture(png("fig_race"), Inches(6.75), Inches(1.45), width=Inches(6.2))
    second = "fig_lm" if os.path.exists(os.path.join(ROOT, "report", "figures", "fig_lm.pdf")) else "fig_fix"
    s6.shapes.add_picture(png(second), Inches(6.75), Inches(3.75), width=Inches(6.2))

    # pictures on slide 5? keep the table. Slide 3 diagram is drawn above.
    prs.save(OUT)
    print("wrote", OUT)


ROLLS = os.environ.get("ROLLS", "[roll numbers]")
LM_SENTENCE = os.environ.get("LM_SENTENCE", "")

if __name__ == "__main__":
    main()
