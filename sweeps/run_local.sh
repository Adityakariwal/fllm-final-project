#!/usr/bin/env bash
# Run a jobs file ("tag: run.py args" per line) in parallel on local CPU cores.
#   sweeps/run_local.sh sweeps/race_jobs.txt results/race 11
set -uo pipefail
cd "$(dirname "$0")/.."
JOBS=$1; OUT=$2; P=${3:-$(($(nproc) - 1))}
PY=${PY:-.venv/bin/python}
mkdir -p "$OUT/logs"
grep -v '^\s*#' "$JOBS" | grep -v '^\s*$' | while IFS= read -r line; do
  tag=${line%%:*}; args=${line#*:}
  [ -f "$OUT/$tag.json" ] && continue                      # resume: skip finished jobs
  echo "$PY src/run.py $args --device cpu --threads 1 --out $OUT/$tag.json > $OUT/logs/$tag.log 2>&1"
done | xargs -P "$P" -I{} bash -c "{}"
echo "all jobs done"
