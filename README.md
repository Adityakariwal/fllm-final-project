# Racing to the Sink: Why Gated Attention Still Forms Attention Sinks, and What Decides the Race

End-term project for **Foundations of Large Language Models**, Group **G28**:
Aditya Kariwal, Manish Kundu, Samyak Sheersh.

Report: [`report/main.pdf`](report/main.pdf) (ICLR 2025 template).

## What this project does

[Ran-Milo (2026)](https://arxiv.org/abs/2603.11487) proved that softmax attention *must* form an
attention sink to implement a trigger-conditional "no-op", and suggested that explicit gating or
non-normalised attention is the way to get sink-free models. We test that suggestion in the paper's own
controlled setting, using our own implementation (not the authors' code).

| | Finding |
|---|---|
| Reproduction | Softmax always forms a sink; ReLU never does (5 seeds, 1-4 layers). |
| Not necessary | Explicit zero-loss, **sink-free** constructions for gated and softmax-off-by-one attention (`src/constructions.py`). |
| ...but learned | Trained gated attention keeps full BOS sinks at every depth (1-8 layers) and every gate initialisation (0.02-0.98). |
| No BOS | Without a BOS token, gated and off-by-one attention use their off switch; softmax fails and forms no substitute anchor, even with positional embeddings. |
| A race | The sink logit (a product q.k) grows ~t^2, the gate logit ~t. Speeding up only the gate (x10) removes every sink in a 4-layer model; speeding up W_Q, W_K as well brings every sink back; a x10 larger learning rate overall keeps the sinks. |
| Theory | A sign-Adam model of the race predicts (P1) a shared log-budget s* + (-z*) = const, (P2) no dependence on the base learning rate, (P3) a critical gate multiplier lambda* ~ sqrt(d_head). P1 (8.8 +- 1.8, r = -0.83) and P2 hold; P3 holds with slope 0.55 but a constant ~5x larger than predicted. SGD and 4-layer width-dependence are where it fails. |
| Gate designs | A sqrt(d_head)-scaled "race-matched" gate is sink-free at d_head = 16/64 and 8 layers, but not at d_head = 4; a bilinear gate gives partial sinks, as predicted. |
| Language model | Small char-level GPT on TinyStories (`src/lm.py`, Kaggle T4): see report Sec. 9. |

## Repository layout

```
src/
  sinks.py          task, attention variants (softmax, softmax1, gated, relu, sigmoid, sigmoidn), metrics, training
  run.py            CLI: train attention types x seeds, dump per-head statistics to JSON
  constructions.py  explicit sink-free solutions for gated / off-by-one attention (Table 2)
  probe.py          gradient probes: pull towards sink vs. towards closing the gate (Fig. 5)
  paper_figs.py     all report figures (PDF) from results/
  plots.py          exploratory per-run attention grids + figures/summary.md
  timeline.py       exploratory training-timeline plot
  race_theory.py    closed-form lambda*(d_head) and a minimal-model Adam/SGD simulation of the race
  lm.py             small character-level GPT on TinyStories (softmax / off-by-one / gated variants)
  summarize.py      recomputes every number quoted in the report -> results/summary.json
sweeps/
  run_local.sh      runs a jobs file ("tag: run.py args") in parallel on local CPU cores (resumable)
  race_jobs*.txt    the race sweeps of Sec. 7-8 (lambda x d_head, base lr, controls, gate designs, SGD)
ppt/
  build_ppt.py      builds G28_presentation.pptx from the course template
  export_pdf.sh     exports it to G28_presentation.pdf with Microsoft PowerPoint (WSL -> Windows)
kaggle/
  build_kernel.py   bundles src/ + a job list into one Kaggle script kernel (2x T4)
  jobs_*.txt        the job lists that were run on Kaggle
results/            raw JSON results of every run (+ logs/); phase2/ original study, race/ race sweeps,
                    theory/ minimal-model simulation, lm/ language model, summary.json all report numbers
report/             LaTeX source (ICLR 2025 style), references, figures, compiled PDF
figures/            exploratory figures
reproduce.sh        every experiment command, and figure regeneration
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing

```bash
./reproduce.sh figures   # regenerate all figures from the saved results (seconds)
./reproduce.sh all       # re-run every experiment (many CPU-hours), then the figures
```

Single experiments, for example:

```bash
# paper reproduction + our variants, 1 layer, 5 seeds
python src/run.py --D 1 --H 1 --lr 1e-3 --seeds 0 1 2 3 4 --out results/1L1H_all.json
# gated attention with a 10x faster gate, 4 layers x 4 heads
python src/run.py --D 4 --H 4 --lr 1e-4 --attn gated --gate_lr_mult 10 --seeds 0 1 2 3 4 --out results/glr10.json
# no BOS token
python src/run.py --D 1 --H 1 --lr 1e-3 --no_bos --steps 20000 --out results/nobos.json
```

Useful flags of `src/run.py`: `--attn` (any of the six variants), `--gate_bias` (initial gate =
sigmoid(bias)), `--gate_lr_mult`, `--qk_lr_mult`, `--opt adam|sgd`, `--dh` (head width),
`--gate_kind linear|scalar|scaled|bilinear`, `--no_bos`, `--pos_emb`, `--residual 0/1`,
`--track_every k` (log per-head statistics during training).

```bash
# critical gate multiplier: 1 layer x 4 heads, head width 64, gate lr x32
python src/run.py --D 1 --H 4 --lr 1e-4 --residual 1 --attn gated --dh 64 --gate_lr_mult 32 --out results/x.json
# race-matched gate in a 4-layer model
python src/run.py --D 4 --H 4 --lr 1e-4 --attn gated --gate_kind scaled --out results/y.json
# all race sweeps on local cores, then every number in the report
sweeps/run_local.sh sweeps/race_jobs.txt results/race 11 && python src/summarize.py
# small language model (GPU)
python src/lm.py --attn gated --gate_lr_mult 10 --out results/lm/lm_gated_lam10_s0.json
```

### Running on Kaggle

The 4-8 layer sweeps were run on Kaggle (2x T4). `kaggle/build_kernel.py` packs `src/` and a job list
into one script kernel that runs the jobs in parallel across the GPUs:

```bash
python kaggle/build_kernel.py --slug my-sweep --jobs-file kaggle/jobs_phase2.txt --user <kaggle-username>
# job lines may name another script, e.g. "lm_softmax_s0: lm.py --attn softmax --seed 0" (kaggle/jobs_lm.txt)
kaggle kernels push -p kaggle/build/my-sweep --accelerator NvidiaTeslaT4
kaggle kernels output <kaggle-username>/my-sweep -p kaggle/out/my-sweep
```

## Settings

Following the original paper and its reference code: token dimension 16, sequence length 16,
attention-only models (no MLP / LayerNorm / positional encoding), Adam (0.9, 0.95), batch 128,
init N(0, 0.02^2), lr 1e-3 (1 layer) or 1e-4 (residual models), stop when the l_inf batch error is below
5e-3. A head is a *sink head* if its mean attention on BOS over non-trigger rows exceeds 0.6. Results are
averaged over 5 seeds (original study) or 3 seeds (race sweeps). GPU and CPU runs of the same seed can differ slightly.

## Reference

The base paper: Yuval Ran-Milo, *Attention Sinks Are Provably Necessary in Softmax Transformers:
Evidence from Trigger-Conditional Tasks*, arXiv:2603.11487, 2026.
