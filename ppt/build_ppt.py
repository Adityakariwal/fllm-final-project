"""Build the presentation from the course template (6 slides, same order, titles and master).

  python ppt/build_ppt.py          # -> ppt/G28_presentation.pptx   (then ppt/export_pdf.sh for the PDF)

Maths is typeset with LaTeX (same fonts as the report) and inserted as transparent 600-dpi images; code is
syntax-coloured with pygments; every number is read from results/summary.json (python src/summarize.py).
Needs: python-pptx, pygments, pdflatex + pdftocairo (TeX Live / poppler).
Environment: ROLLS="..." for the roll numbers on the title slide.
"""
import hashlib
import json
import os
import re
import subprocess
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
from pygments.lexers import PythonLexer
from pygments.token import Token

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
TEMPLATE = os.path.join(ROOT, "ppt", "Project_Presentation_Template.pptx")
OUT = os.path.join(ROOT, "ppt", "G28_presentation.pptx")
ASSETS = os.path.join(ROOT, "ppt", "assets")
S = json.load(open(os.path.join(ROOT, "results", "summary.json")))
ROLLS = os.environ.get("ROLLS", "[roll numbers]")


def rgb(h):
    return RGBColor.from_string(h.lstrip("#").upper())


INK, MUTED, NAVY = "1F1F1F", "52514E", "1B2A5A"
BLUE, GREEN, ORANGE, VIOLET = "2A78D6", "1BAF7A", "EB6834", "6250D6"
TINT = {BLUE: "EAF2FC", GREEN: "E6F6EF", ORANGE: "FDEFE8", VIOLET: "EEECFB", NAVY: "ECEEF4"}


# ============================================================================ LaTeX equations
def eq_png(latex, size=16, color=INK, width_in=None, bold=False):
    """Typeset `latex` (math if no width_in, else a text paragraph of that width) -> (png, w_in, h_in)."""
    body = latex if width_in else f"$\\displaystyle {latex}$"
    if bold:
        body = "\\boldmath\\bfseries " + body
    opts = f"border=1.5pt,varwidth={width_in}in" if width_in else "border=1.5pt"
    src = (f"\\documentclass[{opts}]{{standalone}}\n"
           "\\usepackage{amsmath,amssymb,bm,times,anyfontsize}\n\\usepackage{xcolor}\n"
           "\\begin{document}\n"
           f"\\fontsize{{{size}}}{{{size * 1.3:.1f}}}\\selectfont\\color[HTML]{{{color}}}{body}\n"
           "\\end{document}\n")
    key = hashlib.md5(src.encode()).hexdigest()[:12]
    d = os.path.join(ASSETS, "eq")
    os.makedirs(d, exist_ok=True)
    png = os.path.join(d, key + ".png")
    pdf = os.path.join(d, key + ".pdf")
    if not os.path.exists(png):
        open(os.path.join(d, key + ".tex"), "w").write(src)
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", key + ".tex"], cwd=d,
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"LaTeX failed for {latex!r}:\n" + r.stdout[-1500:])
        subprocess.run(["pdftocairo", "-png", "-r", "600", "-transp", "-singlefile", pdf, png[:-4]], check=True)
    info = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout
    w, h = map(float, re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info).groups())
    return png, w / 72, h / 72


def eq(slide, x, y, latex, size=16, color=INK, width_in=None, anchor="tl", bold=False):
    """Place an equation at (x, y) [inches]; anchor 'tl' = top-left, 'c' = centre. Returns (w, h)."""
    png, w, h = eq_png(latex, size, color, width_in, bold)
    if anchor == "c":
        x, y = x - w / 2, y - h / 2
    slide.shapes.add_picture(png, Inches(x), Inches(y), Inches(w), Inches(h))
    return w, h


# ============================================================================ shapes and text
def rect(slide, x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None, lw=0.75):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill:
        sh.fill.solid()
        sh.fill.fore_color.rgb = rgb(fill)
    else:
        sh.fill.background()
    if line:
        sh.line.color.rgb = rgb(line)
        sh.line.width = Pt(lw)
    else:
        sh.line.fill.background()
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
    sh.shadow.inherit = False
    return sh


def text(slide, x, y, w, h, runs, size=14, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space=4):
    """runs: str | list of paragraphs; a paragraph is a str or a list of (text, bold[, color])."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = 1.08
        p.space_after = Pt(space)
        for item in ([(para, False)] if isinstance(para, str) else para):
            r = p.add_run()
            r.text = item[0]
            r.font.size = Pt(size)
            r.font.bold = item[1]
            r.font.color.rgb = rgb(item[2] if len(item) > 2 else color)
    return tb


def callout(slide, x, y, w, h, accent):
    """Tinted panel with a coloured accent bar on the left."""
    rect(slide, x, y, w, h, fill=TINT[accent])
    rect(slide, x, y, 0.07, h, fill=accent)


def arrow(slide, x0, y0, x1, y1):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x0), Inches(y0), Inches(x1), Inches(y1))
    c.line.color.rgb = rgb(MUTED)
    c.line.width = Pt(1.5)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle"}))


def stat_card(slide, x, y, w, h, value, caption, accent, vsize=24):
    rect(slide, x, y, w, h, fill=TINT[accent])
    rect(slide, x, y, w, 0.06, fill=accent)
    text(slide, x + 0.12, y + 0.12, w - 0.24, 0.55, [[(value, True, accent)]], size=vsize)
    text(slide, x + 0.12, y + 0.66, w - 0.24, h - 0.7, caption, size=11.5, color=MUTED)


def table(slide, x, y, w, col_w, rows, size=12, highlight=()):
    tb = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w),
                                Inches(0.34 * len(rows))).table
    for j, cw in enumerate(col_w):
        tb.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tb.rows[i].height = Inches(0.33)
        for j, v in enumerate(row):
            cell = tb.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(NAVY if i == 0 else (TINT[GREEN] if i in highlight else
                                                                ("F7F7F5" if i % 2 else "FFFFFF")))
            cell.text = v
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(size)
                    r.font.bold = i == 0
                    r.font.color.rgb = rgb("FFFFFF" if i == 0 else INK)
    return tb


def remove(shape):
    shape._element.getparent().remove(shape._element)


def body(slide):
    """The template's free text box below the title (not a numbered placeholder)."""
    return [sh for sh in slide.shapes if sh.has_text_frame and sh.left < Inches(1.5) and sh.top > Inches(1)][0]


# ============================================================================ syntax-coloured code
ONE_LIGHT = [(Token.Comment, "A0A1A7", True), (Token.Keyword, "A626A4", False),
             (Token.Name.Builtin.Pseudo, "E45649", False), (Token.Name.Builtin, "C18401", False),
             (Token.Name.Function, "4078F2", False), (Token.Name.Class, "4078F2", False),
             (Token.Literal.String, "50A14F", False), (Token.Literal.Number, "986801", False),
             (Token.Operator, "0184BC", False)]


def style(tok):
    for t, col, it in ONE_LIGHT:
        if tok in t:
            return col, it
    return "383A42", False


def code_block(slide, x, y, w, title, src, size=11):
    lines = src.strip("\n").split("\n")
    text(slide, x, y, w, 0.3, [[(title, True, MUTED)]], size=11)
    h = len(lines) * size * 1.2 / 72 + 0.2
    sh = rect(slide, x, y + 0.32, w, h, fill="FAFAFA", line="E3E2DE")
    tf = sh.text_frame
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_right = Inches(0.12)
    tf.margin_top = tf.margin_bottom = Inches(0.08)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    p.line_spacing = 1.0
    for tok, val in PythonLexer().get_tokens("\n".join(lines)):
        for k, piece in enumerate(val.split("\n")):
            if k > 0:
                p = tf.add_paragraph()
                p.alignment = PP_ALIGN.LEFT
                p.line_spacing = 1.0
            if piece:
                col, it = style(tok)
                r = p.add_run()
                r.text = piece
                r.font.name = "Consolas"
                r.font.size = Pt(size)
                r.font.italic = it
                r.font.color.rgb = rgb(col)
    return y + 0.32 + h


# ============================================================================ numbers
def race_numbers():
    import numpy as np
    import paper_figs as P
    from race_theory import lam_star
    lams = [1, 2, 4, 8, 16, 32, 64, 128]
    real = {}
    for dh in [4, 8, 16, 32]:
        m = [np.mean([P.mean_bos(r) for r in P.race(f"dh{dh}_lam{l}_s*")]) if P.race(f"dh{dh}_lam{l}_s*")
             else np.nan for l in lams]
        real[dh] = P.cross(lams, m)
    d, v = np.array(list(real)), np.array(list(real.values()))
    slope = np.polyfit(np.log(d), np.log(v), 1)[0]
    offset = float(np.exp(np.mean(np.log(v / np.array([lam_star(x) for x in d])))))
    return real, slope, offset


# ============================================================================ slides
def slide_title(s):
    title, sub = [sh for sh in s.shapes if sh.is_placeholder][:2]
    title.text_frame.text = "Racing to the Sink: Why Gated Attention Still Forms Attention Sinks"
    for r in title.text_frame.paragraphs[0].runs:
        r.font.size = Pt(30)
    tf = sub.text_frame
    tf.clear()
    lines = [[("Attention sinks are necessary for softmax, and preferred by training", False, BLUE)],
             [("Group Number: ", True), ("G28", False)],
             [("Names: ", True), ("Aditya Kariwal, Manish Kundu, Samyak Sheersh", False)],
             [("Roll Numbers: ", True), (ROLLS, False)],
             [("Base paper: ", True), ("Ran-Milo (2026), Attention Sinks Are Provably Necessary in Softmax "
                                       "Transformers, arXiv:2603.11487", False)]]
    for i, para in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(6 if i else 12)
        p.line_spacing = 1.0
        for t, b, *c in para:
            r = p.add_run()
            r.text = t
            r.font.size = Pt(17 if i == 0 else 15)
            r.font.bold = b
            r.font.italic = i == 0
            r.font.color.rgb = rgb(c[0] if c else INK)


def slide_problem(s):
    remove(body(s))
    # token-sequence diagram: input row and target row
    X0, Y0, BW, BH, G = 1.6, 1.4, 0.8, 0.5, 0.08
    cells = [(r"\textsc{bos}", NAVY), ("x_1", "DCE1EA"), ("x_2", "DCE1EA"), (r"\cdots", None),
             ("x_j", ORANGE), (r"\cdots", None), ("x_{15}", "DCE1EA")]
    tgt = ["0", "0", "0", r"\cdots", None, r"\cdots", "0"]
    text(s, 0.6, Y0 + 0.1, 1.0, 0.35, [[("input", True, MUTED)]], size=13)
    y2 = Y0 + BH + 0.3
    text(s, 0.6, y2 + 0.1, 1.0, 0.35, [[("target", True, MUTED)]], size=13)
    for k, (lab, col) in enumerate(cells):
        x = X0 + k * (BW + G)
        if col:
            dark = col in (NAVY, ORANGE)
            rect(s, x, Y0, BW, BH, fill=col)
            eq(s, x + BW / 2, Y0 + BH / 2, lab, size=16, color="FFFFFF" if dark else INK, anchor="c", bold=dark)
        else:
            eq(s, x + BW / 2, Y0 + BH / 2, lab, size=15, color=MUTED, anchor="c")
        if tgt[k] == r"\cdots":
            eq(s, x + BW / 2, y2 + BH / 2, tgt[k], size=15, color=MUTED, anchor="c")
        elif tgt[k] is not None:
            rect(s, x, y2, BW, BH, fill="EFEFEC")
            eq(s, x + BW / 2, y2 + BH / 2, tgt[k], size=15, color=INK, anchor="c")
        else:   # the trigger's target: mean of the prefix
            rect(s, x - 0.25, y2, BW + 0.5, BH, fill=TINT[ORANGE], line=ORANGE)
            eq(s, x + BW / 2, y2 + BH / 2, r"\mathrm{mean}(x_{1..j})", size=13, anchor="c")
    text(s, X0, y2 + BH + 0.12, 6.0, 0.35,
         "the trigger sits at a random position j; every token carries 13 random content features",
         size=11.5, color=MUTED)
    # task equations
    eq(s, 8.0, 1.3, r"y^{(i)}=\begin{cases}\dfrac{1}{j}\sum_{k=1}^{j}x^{(k)} & i=j\quad(\text{trigger})\\[6pt]"
                    r"\;0 & i\neq j\quad(\text{no-op})\end{cases}", size=17)
    eq(s, 8.0, 2.72, r"\mathcal{L}(f)=\max_i\big\|y^{(i)}-f(x)^{(i)}\big\|_\infty<5\cdot10^{-3}", size=15)
    # theorem callout
    callout(s, 0.6, 3.5, 12.2, 1.18, BLUE)
    text(s, 0.85, 3.57, 4.0, 0.38, [[("Theorem (Ran-Milo, 2026)", True, BLUE)]], size=14)
    eq(s, 0.85, 4.0, r"\mathcal{L}(f)\le\eta\;\Longrightarrow\;\alpha_{i,\textsc{bos}}\ge 1-\varepsilon"
                     r"\ \text{ at every non-trigger row } i", size=16)
    text(s, 7.7, 3.62, 4.95, 1.0,
         ["A softmax model needs an attention sink to do nothing; ReLU attention does not.",
          [("The paper's remedy: gating or un-normalised attention.", True)]], size=13)
    # our question
    callout(s, 0.6, 4.88, 12.2, 1.18, ORANGE)
    text(s, 0.85, 4.95, 4.0, 0.38, [[("Our question", True, ORANGE)]], size=14)
    text(s, 0.85, 5.35, 11.8, 0.7,
         "A gated head can output zero without any sink (we construct one). Does training actually use the "
         "gate? If not, what decides which way a head switches off, and can we control it?", size=14)
    text(s, 0.6, 6.3, 12.2, 0.6,
         [[("Data pipeline.  ", True, MUTED), ("Synthetic task: 128 fresh sequences per step, generated on the "
                                              "fly.  Language model: TinyStories at character level, "
                                              "[BOS] + 255 characters per sample.", False, MUTED)]], size=12.5)


def slide_arch(s):
    remove(body(s))
    text(s, 0.6, 1.2, 6.5, 0.4, [[("Attention-only transformer (as in the base paper)", True, NAVY)]], size=15)
    eq(s, 0.6, 1.62, r"D\text{ layers}\times H\text{ heads of width }d_h,\ \text{residual for }D>1,"
                     r"\ \text{no MLP / LayerNorm / positions}", size=12.5, color=MUTED)
    eq(s, 0.7, 2.05, r"\begin{aligned}"
                     r"\text{head}(x)_i&=W_O\Big(g_i\odot\textstyle\sum_{k\le i}\alpha_{ik}\,W_Vx_k\Big),"
                     r"\quad s_{ik}=q_i^{\top}k_k\\[4pt]"
                     r"\text{softmax:}\quad\alpha_{ik}&=e^{s_{ik}}\big/\textstyle\sum_{k'}e^{s_{ik'}}\\[3pt]"
                     r"\text{off-by-one:}\quad\alpha_{ik}&=e^{s_{ik}}\big/\big(1+\textstyle\sum_{k'}e^{s_{ik'}}\big)"
                     r"\\[3pt]"
                     r"\text{gated (Qiu et al.):}\quad g_i&=\sigma(W_Gx_i+b)\end{aligned}", size=15)
    callout(s, 0.6, 4.3, 6.5, 1.6, GREEN)
    text(s, 0.85, 4.36, 6.0, 0.38, [[("Gate parameterisations we study", True, GREEN)]], size=14)
    eq(s, 0.95, 4.78, r"\begin{aligned}"
                      r"\text{linear}+\lambda:&\quad \mathrm{lr}(W_G,b)=\lambda\cdot\mathrm{lr}\\[1pt]"
                      r"\text{race-matched:}&\quad g_i=\sigma\big(\kappa(W_Gx_i+b)\big),\ \kappa=2.5\sqrt{d_h}\\[1pt]"
                      r"\text{bilinear:}&\quad z_i=A_h(B_hx_i)+b\end{aligned}", size=14)
    text(s, 0.6, 6.1, 6.5, 0.8,
         [[("Language model:  ", True, MUTED), ("character-level GPT, 6 layers, d = 384, 6 heads of width 64, "
                                               "pre-LN + MLP, context 256, AdamW; same attention variants.",
                                               False, MUTED)]], size=12.5)
    # two routes to "off"
    X = 7.45
    text(s, X, 1.2, 5.4, 0.4, [[("Two ways for a head to switch off", True, NAVY)]], size=15)
    rect(s, X + 1.3, 1.72, 2.8, 0.6, fill="EFEFEC", line="8A8984", shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.15, lw=1.5)
    eq(s, X + 2.7, 2.02, r"\text{non-trigger row: output }0", size=15, anchor="c")
    for bx, col, title, l1, l2 in [
            (X, BLUE, r"\textbf{sink route}", r"s=\textstyle\sum_{r\le d_h}Q_rK_r", r"s\approx d_h(\eta t)^2"),
            (X + 2.95, GREEN, r"\textbf{gate route}", r"z=W_Gx+b", r"-z\approx 2\lambda\,\eta t")]:
        rect(s, bx, 2.92, 2.45, 1.28, fill=TINT[col], line=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
             radius=0.1, lw=1.75)
        eq(s, bx + 1.225, 3.17, title, size=15, color=col, anchor="c")
        eq(s, bx + 1.225, 3.56, l1, size=15, anchor="c")
        eq(s, bx + 1.225, 3.95, l2, size=15, anchor="c")
    rect(s, X + 0.4, 4.75, 4.6, 0.72, fill=TINT[ORANGE], line=ORANGE, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.15, lw=1.75)
    eq(s, X + 2.7, 5.11, r"\text{shared budget:}\ \ s^*+(-z^*)\approx T'=\log\frac{C\,i}{\varepsilon}",
       size=15, anchor="c")
    arrow(s, X + 2.25, 2.32, X + 1.3, 2.92)
    arrow(s, X + 3.15, 2.32, X + 4.15, 2.92)
    arrow(s, X + 1.3, 4.2, X + 2.1, 4.75)
    arrow(s, X + 4.15, 4.2, X + 3.3, 4.75)
    text(s, X, 5.62, 5.4, 0.4, "The faster route takes the larger share of the budget, which gives",
         size=12.5, color=MUTED)
    eq(s, X + 2.7, 6.42, r"\lambda^*=\frac{T'-S_0}{2\sqrt{S_0}}\,\sqrt{d_h}", size=18, anchor="c")


def slide_code(s):
    remove(body(s))
    text(s, 0.6, 1.2, 12.2, 0.35,
         [[("Written from scratch in PyTorch.  ", True, MUTED),
           ("Excerpts of our contributions; full code in src/.", False, MUTED)]], size=13)
    W, Z = 5.95, 12
    yA = code_block(s, 0.6, 1.62, W, "src/sinks.py  ·  gate parameterisations", '''
if gate_kind == "bilinear":           # z = A_h (B_h x) + b
    self.W_GB = nn.Linear(d, H * dh, bias=False)
    self.W_GA = nn.Parameter(torch.empty(H, dh, dh))
else:
    self.W_G = nn.Linear(d, H * dh, bias=True)
    if gate_kind == "scaled":         # race-matched gate
        self.kappa = gate_scale * math.sqrt(dh)
self.W_G.weight.data.div_(self.kappa) # muP: init / kappa

def gate_logits(self, x):            # == gate lr x kappa
    return self.kappa * self.W_G(x)
''', size=Z)
    yB = code_block(s, 6.85, 1.62, W, "src/sinks.py  ·  racing the two routes", '''
gate_p = [p for n, p in named if "W_G" in n]
qk_p = [p for n, p in named if n[:3] in ("W_Q", "W_K")]
groups = [
    dict(params=other_p, lr=lr),
    dict(params=qk_p, lr=lr * qk_lr_mult),     # sink
    dict(params=gate_p, lr=lr * gate_lr_mult), # gate
]
opt = torch.optim.Adam(groups, betas=(0.9, 0.95))
# stop when the l_inf error on a batch < 5e-3
''', size=Z)
    y2 = max(yA, yB) + 0.3
    code_block(s, 0.6, y2, W, "src/race_theory.py  ·  closed-form critical multiplier", '''
def lam_star(dh, i=8, eps=5e-3, vbar=1.0, a0=0.6):
    T = math.log(vbar * i / eps)       # shared budget T'
    S0 = math.log(i * a0 / (1 - a0))   # sink-head logit
    return (T - S0) * math.sqrt(dh) / (2 * math.sqrt(S0))
''', size=Z)
    code_block(s, 6.85, y2, W, "src/probe.py  ·  zero-valued gradient probes", '''
p_s = torch.zeros(H, requires_grad=True)  # BOS logit
p_g = torch.zeros(H, requires_grad=True)  # gate logit
layer.probe = (row_mask, p_s, p_g)        # non-trigger rows
gs = torch.autograd.grad(loss, ps + pg)
gate_pull = g / layer.dh                  # per coordinate
''', size=Z)


def slide_eval(s):
    remove(body(s))
    text(s, 0.6, 1.2, 6.6, 0.4, [[("Methods compared", True, NAVY)]], size=15)
    table(s, 0.6, 1.62, 6.6, [2.35, 2.6, 1.65], [
        ("Method", "Role", "Source"),
        ("softmax", "baseline: sink is necessary", "Ran-Milo 2026"),
        ("ReLU", "sink-free, un-normalised", "Ran-Milo 2026"),
        ("softmax-off-by-one", "null slot: can switch off", "Miller 2023"),
        ("linear output gate", "gate: can switch off", "Qiu et al. 2025"),
        ("gate + lr multiplier λ", "speeds up the gate route", "ours"),
        ("race-matched gate", "κ = 2.5·√d_h, muP-style", "ours"),
        ("bilinear gate", "gate as fast as the sink", "ours"),
        ("controls", "Q/K-lr ×10, all-lr ×10, SGD", "ours"),
    ], size=12, highlight=(5, 6, 7, 8))
    text(s, 7.55, 1.2, 5.3, 0.4, [[("Evaluation metrics", True, NAVY)]], size=15)
    table(s, 7.55, 1.62, 5.25, [1.45, 3.8], [
        ("Metric", "Definition"),
        ("BOS mass", "mean attention on BOS, non-trigger rows"),
        ("sink head", "BOS mass > 0.6"),
        ("gate value", "importance-weighted, off / trigger rows"),
        ("steps", "until ℓ∞ error < 5·10⁻³"),
        ("s*, −z*", "sink-logit gap / gate closing at stop"),
        ("λ*", "λ where mean BOS mass crosses 0.6"),
        ("LM", "val loss, BOS attention, BOS outlier"),
    ], size=12)
    cards = [("3–5", "seeds per configuration", BLUE), ("558", "models trained in total", GREEN),
             ("11 CPU + 2×T4", "local cores and Kaggle GPUs", VIOLET),
             ("1 script", "src/summarize.py recomputes every number", ORANGE)]
    for k, (v, c, col) in enumerate(cards):
        stat_card(s, 0.6 + k * 3.08, 5.25, 2.92, 1.2, v, c, col)


def slide_results(s):
    remove(body(s))
    real, slope, offset = race_numbers()
    c = S["ctl_lam10"], S["ctl_lam10_qk10"]
    lr = [S[k]["mean_bos"] for k in ("lr3e-5_lam16", "dh16_lam16", "lr3e-4_lam16")]
    cards = [(f"{c[0]['sinks']}/{c[0]['heads']} → {c[1]['sinks']}/{c[1]['heads']}",
              "sink heads with gate lr ×10, then also Q/K lr ×10: the race is causal", BLUE),
             (f"r = {S['budget']['corr']:.2f}",
              f"shared log-budget: s* + (−z*) = {S['budget']['mean']:.1f} ± {S['budget']['sd']:.1f}", GREEN),
             (f"slope {slope:.2f}", f"λ* ∝ √d_h (theory 0.5); constant ×{offset:.1f}", VIOLET),
             (" · ".join(f"{v:.2f}" for v in lr), "BOS mass at λ = 16 for base lr 3e-5 / 1e-4 / 3e-4: identical", ORANGE)]
    for k, (v, cap, col) in enumerate(cards):
        stat_card(s, 0.6 + k * 3.08, 1.2, 2.92, 1.22, v, cap, col)
    os.makedirs(ASSETS, exist_ok=True)
    fig = os.path.join(ASSETS, "fig_race")
    subprocess.run(["pdftoppm", "-png", "-r", "250", "-singlefile",
                    os.path.join(ROOT, "report", "figures", "fig_race.pdf"), fig], check=True)
    s.shapes.add_picture(fig + ".png", Inches(0.6), Inches(2.62), width=Inches(7.6))
    rows = [("4 layers × 4 heads, 3 seeds", "sink heads", "steps")]
    for tag, lab in [("ctl_base_lam1", "baseline, λ = 1"), ("ctl_lam10", "gate lr ×10"),
                     ("ctl_lam10_qk10", "gate ×10 and Q/K ×10"), ("ctl_lr1e-3", "all learning rates ×10")]:
        r = S[tag]
        rows.append((lab, f"{r['sinks']}/{r['heads']}", f"{r['steps'] / 1e3:.1f}k"))
    table(s, 0.6, 5.12, 7.6, [4.4, 1.6, 1.6], rows, size=11.5)
    text(s, 8.5, 2.62, 4.3, 4.4, [
        [("What holds.  ", True, GREEN), ("Sinks are preferred, not only necessary: gated models keep them at "
                                          "every depth and gate init, though a sink-free solution exists.", False)],
        [("What fails.  ", True, ORANGE), ("SGD still forms sinks; at 4 layers λ* hardly depends on width, so "
                                            "the √d_h gate fails at d_h = 4.", False)],
        [("Limitations.  ", True, MUTED), ("Synthetic task; first-order theory (constant off by ~5×); our "
                                           "small LMs formed no sinks in 20k steps, so the LM test is open.",
                                           False)],
        [("Conclusion.  ", True, NAVY), ("An off switch makes sinks unnecessary but not absent: the gate must "
                                         "also learn fast enough. Its learning rate decides the race.", False)],
    ], size=13, space=10)


def main():
    prs = Presentation(TEMPLATE)
    s1, s2, s3, s4, s5, s6 = prs.slides
    slide_title(s1)
    slide_problem(s2)
    slide_arch(s3)
    slide_code(s4)
    slide_eval(s5)
    slide_results(s6)
    prs.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
