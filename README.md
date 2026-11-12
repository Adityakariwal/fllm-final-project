# Necessary, but Also Preferred: Why Attention Sinks Form Even When Transformers Can Switch Off

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
| Not necessary | We give explicit zero-loss, **sink-free** weight constructions for gated and softmax-off-by-one attention (`src/constructions.py`). |
| ...but learned | Trained gated attention still puts >= 0.95 attention on BOS at every depth (1-8 layers) and for every gate initialisation (0.02-0.98). Its gate stays near its initial value. |
| Depth | Softmax-off-by-one moves from BOS sinks to its null slot only as depth grows (7/160 sink heads at 8 layers). |
| No BOS | Without a BOS token, gated and off-by-one attention *do* use their off switch; softmax fails and forms no substitute anchor. |
| Mechanism | Gradient probes show the loss pulls **harder** towards closing the gate than towards the sink, yet the sink wins a **speed race**: the bilinear q·k sink grows faster than the linear gate. |
| Intervention | Raising only the gate's learning rate by 10x removes **all** sinks in a 4-layer model and trains **5x faster**. |

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
kaggle/
  build_kernel.py   bundles src/ + a job list into one Kaggle script kernel (2x T4)
  jobs_*.txt        the job lists that were run on Kaggle
results/            raw JSON results of every run (+ logs/); phase2/ holds Sec. 5-7 experiments
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
sigmoid(bias)), `--gate_lr_mult`, `--no_bos`, `--residual 0/1`, `--track_every k` (log per-head
statistics during training).

### Running on Kaggle

The 4-8 layer sweeps were run on Kaggle (2x T4). `kaggle/build_kernel.py` packs `src/` and a job list
into one script kernel that runs the jobs in parallel across the GPUs:

```bash
python kaggle/build_kernel.py --slug my-sweep --jobs-file kaggle/jobs_phase2.txt --user <kaggle-username>
kaggle kernels push -p kaggle/build/my-sweep --accelerator NvidiaTeslaT4
kaggle kernels output <kaggle-username>/my-sweep -p kaggle/out/my-sweep
```

## Settings

Following the original paper and its reference code: token dimension 16, sequence length 16,
attention-only models (no MLP / LayerNorm / positional encoding), Adam (0.9, 0.95), batch 128,
init N(0, 0.02^2), lr 1e-3 (1 layer) or 1e-4 (residual models), stop when the l_inf batch error is below
5e-3. A head is a *sink head* if its mean attention on BOS over non-trigger rows exceeds 0.6. All results
are averaged over 5 seeds. GPU and CPU runs of the same seed can differ slightly.

## Reference

The base paper: Yuval Ran-Milo, *Attention Sinks Are Provably Necessary in Softmax Transformers:
Evidence from Trigger-Conditional Tasks*, arXiv:2603.11487, 2026.
