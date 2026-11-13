"""Bundle src/*.py into a single Kaggle script kernel that runs a list of jobs in parallel.

Usage:
  python kaggle/build_kernel.py --slug sinks-phase1-4l4h --jobs-file kaggle/jobs_4l4h.txt
  kaggle kernels push -p kaggle/build/sinks-phase1-4l4h --accelerator NvidiaTeslaT4

Each line of the jobs file is "tag: [script.py] args" -- the argument string for src/run.py (default) or
for another script in src/ such as lm.py (without --out/--device); jobs are round-robined across the
visible GPUs. Outputs land in /kaggle/working/results. If any job uses lm.py, the TinyStories data is
downloaded once before the jobs start (into /tmp/data, so it is not part of the kernel output).
"""
import argparse
import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

LAUNCHER = r'''
import os, subprocess, sys, pathlib, time, torch
SRC = {src!r}
JOBS = {jobs!r}
W = pathlib.Path("/kaggle/working"); (W / "src").mkdir(exist_ok=True, parents=True)
(W / "results" / "logs").mkdir(parents=True, exist_ok=True)
for name, code in SRC.items():
    (W / "src" / name).write_text(code)
ngpu = max(torch.cuda.device_count(), 1)
if any(a.split()[0] == "lm.py" for _, a in JOBS):
    subprocess.run([sys.executable, str(W / "src" / "lm.py"), "--prefetch_only", "--data_dir", "/tmp/data",
                    "--out", "/tmp/prefetch.json"], check=True)
print("GPUs:", ngpu, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
procs = []
for k, (tag, args) in enumerate(JOBS):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k % ngpu))
    log = open(W / "results" / "logs" / f"{{tag}}.log", "w")
    parts = args.split()
    script = parts.pop(0) if parts and parts[0].endswith(".py") else "run.py"
    extra = ["--data_dir", "/tmp/data"] if script == "lm.py" else []
    cmd = [sys.executable, str(W / "src" / script), *parts, *extra, "--out", str(W / "results" / f"{{tag}}.json")]
    procs.append((tag, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))
t0 = time.time()
while any(p.poll() is None for _, p in procs):
    time.sleep(60)
    print(f"[{{time.time()-t0:.0f}}s] running:", [t for t, p in procs if p.poll() is None], flush=True)
    for t, p in procs:   # latest progress line of every job
        lines = (W / "results" / "logs" / f"{{t}}.log").read_text().strip().splitlines()
        print(f"    {{t}}: {{lines[-1][:150] if lines else '(no output yet)'}}", flush=True)
for tag, p in procs:
    print("=" * 20, tag, "exit", p.returncode)
    print((W / "results" / "logs" / f"{{tag}}.log").read_text()[-3000:])
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--jobs-file", required=True)
    ap.add_argument("--user", default="mk1942")
    a = ap.parse_args()

    src = {p.name: p.read_text() for p in (ROOT / "src").glob("*.py")}
    jobs = []
    for line in pathlib.Path(a.jobs_file).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tag, args = line.split(":", 1)
            jobs.append((tag.strip(), args.strip()))

    out = ROOT / "kaggle" / "build" / a.slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "main.py").write_text(LAUNCHER.format(src=src, jobs=jobs))
    meta = dict(id=f"{a.user}/{a.slug}", title=a.slug, code_file="main.py", language="python",
                kernel_type="script", is_private=True, enable_gpu=True, enable_tpu=False,
                enable_internet=True, dataset_sources=[], competition_sources=[],
                kernel_sources=[], model_sources=[])
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"built {out} with {len(jobs)} jobs")


if __name__ == "__main__":
    main()
