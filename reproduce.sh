#!/usr/bin/env bash
# Reproduce every result and figure in the report.
#
#   ./reproduce.sh figures   # regenerate all figures/tables from saved results/ (seconds)
#   ./reproduce.sh all       # re-run every experiment, then the figures (many CPU-hours;
#                            # the 4-8 layer sweeps were run on Kaggle 2xT4, see kaggle/)
#
# Each block mirrors one experiment in the report. Commands are exactly those used.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
RUN="$PY src/run.py"
SEEDS="0 1 2 3 4"
TYPES="softmax relu softmax1 sigmoid sigmoidn gated"
mkdir -p results/phase2

experiments() {
  # --- Sec. 4 / Table 1: reproduction + variants (1L1H, 2L2H, 4L4H) -------------------
  for t in $TYPES; do
    $RUN --D 1 --H 1 --lr 1e-3 --seeds $SEEDS --attn $t --out results/1L1H_$t.json
    $RUN --D 2 --H 2 --lr 1e-4 --seeds $SEEDS --attn $t --out results/2L2H_$t.json
    $RUN --D 4 --H 4 --lr 1e-4 --seeds $SEEDS --attn $t --out results/4L4H_$t.json
  done

  # --- Sec. 5 / Table 2: explicit sink-free constructions ------------------------------
  $PY src/constructions.py | tee results/phase2/constructions.txt

  # --- Sec. 6 / Fig. 3: depth sweep (H=4, residual; D=4 comes from 4L4H above) ----------
  for t in softmax softmax1 gated; do
    for D in 1 2 3; do
      $RUN --D $D --H 4 --lr 1e-4 --residual 1 --seeds $SEEDS --attn $t --track_every 1000 \
           --out results/phase2/D${D}H4_$t.json
    done
    for D in 6 8; do
      $RUN --D $D --H 4 --lr 1e-4 --steps 80000 --seeds $SEEDS --attn $t --track_every 1000 \
           --out results/phase2/D${D}H4_$t.json
    done
  done

  # --- Sec. 6 / Fig. 2: gate initialisation (bias 0 = 4L4H_gated above) ----------------
  for b in m4:-4 m2:-2 p2:2 p4:4; do
    $RUN --D 4 --H 4 --lr 1e-4 --seeds $SEEDS --attn gated --gate_bias ${b#*:} --track_every 500 \
         --out results/phase2/G4L4Hb${b%%:*}_gated.json
  done

  # --- Sec. 6 / Table 3: no BOS token ----------------------------------------------------
  for t in $TYPES; do
    $RUN --D 1 --H 1 --lr 1e-3 --steps 20000 --no_bos --seeds $SEEDS --attn $t \
         --out results/phase2/noBOS1L1H_$t.json
  done
  for cfg in "2 2" "4 4"; do
    set -- $cfg
    for t in softmax softmax1 gated; do
      $RUN --D $1 --H $2 --lr 1e-4 --no_bos --seeds $SEEDS --attn $t --track_every 1000 \
           --out results/phase2/noBOS$1L$2H_$t.json
    done
    $RUN --D $1 --H $2 --lr 1e-4 --no_bos --seeds $SEEDS --attn relu sigmoidn \
         --out results/phase2/noBOS$1L$2H_relu.json
  done

  # --- Sec. 7 / Fig. 4: timelines (seed 0 with tracking, 4L4H) -------------------------
  $RUN --D 4 --H 4 --lr 1e-4 --attn softmax --seeds 0 --track_every 500 --out results/phase2/D4H4_softmax.json
  $RUN --D 4 --H 4 --lr 1e-4 --attn gated --seeds 0 --track_every 500 --out results/phase2/G4L4Hbp0_gated.json

  # --- Sec. 7 / Fig. 5: gradient probes ------------------------------------------------
  $PY src/probe.py --D 1 --H 1 --lr 1e-3 --every 25 --out results/phase2/probe_1L1H.json

  # --- Sec. 7 / Fig. 6 + Table 4: gate learning-rate multiplier -------------------------
  for m in 1 3 10 30 100; do
    $RUN --D 1 --H 1 --lr 1e-3 --attn gated --gate_lr_mult $m --seeds $SEEDS --steps 20000 \
         --out results/phase2/glr${m}_1L1H_gated.json
  done
  for m in 3 10 30 100; do
    $RUN --D 4 --H 4 --lr 1e-4 --attn gated --gate_lr_mult $m --seeds $SEEDS --track_every 500 \
         --out results/phase2/glr${m}_4L4H_gated.json
  done
  $PY src/probe.py --D 1 --H 1 --lr 1e-3 --every 25 --gate_lr_mult 10 --out results/phase2/probe_1L1H_lam10.json

  # --- Sec. 8: race theory (closed form + minimal-model simulation) ---------------------
  $PY src/race_theory.py --out results/theory/race_sim.json

  # --- Sec. 8-9: race sweeps (lambda* vs head width / lr, controls, race-matched gate,
  #     no-BOS + positional embedding). One run per line, run in parallel on local cores.
  sweeps/run_local.sh sweeps/race_jobs.txt results/race "$(($(nproc) - 1))"
  sweeps/run_local.sh sweeps/race_jobs_sgd.txt results/race 1

  # --- Sec. 10: small language model (TinyStories, Kaggle 2xT4) --------------------------
  #   python kaggle/build_kernel.py --slug sinks-race-lm --jobs-file kaggle/jobs_lm.txt --user <you>
  #   kaggle kernels push -p kaggle/build/sinks-race-lm --accelerator NvidiaTeslaT4
  #   kaggle kernels output <you>/sinks-race-lm -p kaggle/out/sinks-race-lm   # -> copy results/ to results/lm
  # or locally on a GPU, e.g.:
  #   $PY src/lm.py --attn gated --gate_kind scaled --seed 0 --out results/lm/lm_gated_scaled_s0.json
}

figures() {
  $PY src/paper_figs.py --out report/figures   # all report figures (PDF)
  $PY src/plots.py --results results --out figures   # per-run head grids + figures/summary.md
  $PY src/timeline.py --out figures
}

case "${1:-figures}" in
  all) experiments; figures ;;
  figures) figures ;;
  *) echo "usage: $0 [figures|all]"; exit 1 ;;
esac
