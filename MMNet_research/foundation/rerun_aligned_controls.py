"""Re-run the manuscript's controlled analyses on the epoch-aligned caches.

The frozen-encoder controls (joint reference, single-task models, modality-ablation grid) and
the five-signal Huttunen et al. port were run before build_aligned_caches.py fixed the
cardiorespiratory timeline. This driver re-runs them on data/aligned/ and writes every result
beside the earlier one with the suffix "_aligned", so nothing is overwritten:

    runs/final/final_model_aligned.json       joint frozen-encoder model, 3 seeds    (~20 min)
    runs/final/single_task_aligned.json       staging-only, respiratory-only, 3 seeds (~35 min)
    runs/final/ablation_shards_aligned/       9 modality removals x 3 seeds          (~4 h, 2 workers)
    runs/final/huttunen5_aligned.json         Huttunen et al., five signals, 3 seeds (~2 h, estimate)

Times are from the September runs on the RTX 2060 / 3060 Ti machines. Then
summarize_aligned_controls.py prints the numbers for the manuscript tables.

Requires the aligned caches (preprocessing/build_aligned_caches.py) and the frozen LaBraM
cache the earlier runs used.

    python rerun_aligned_controls.py                      # everything, in order
    python rerun_aligned_controls.py --only ablation      # one step: final, single, ablation, huttunen, summary
    python rerun_aligned_controls.py --workers 3          # ablation workers (2 used in September)
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(REPO, "data")
LOG = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final", "rerun_aligned.log")

ENV = {
    "MMNET_FEATURES": "aligned/mm_features",
    "MMNET_MULTIMODAL": "aligned/multimodal",
    "MMNET_PFLOW": "aligned/pressure_flow",
    "MMNET_TAG": "_aligned",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
}
STEPS = ["final", "single", "ablation", "huttunen", "summary"]


def preflight():
    """Stop before any training if the aligned caches are missing or incomplete."""
    ok = True
    for sub in ("mm_features", "multimodal", "pressure_flow"):
        d = os.path.join(DATA, "aligned", sub)
        n = len([f for f in os.listdir(d) if f.endswith(".npz")]) if os.path.isdir(d) else 0
        print("  data/aligned/%-14s %3d recordings" % (sub, n))
        ok &= n >= 99
    if not ok:
        sys.exit("aligned caches incomplete: run preprocessing/build_aligned_caches.py first")


def run(cmd, extra_env=None):
    env = dict(os.environ, **ENV, **(extra_env or {}))
    t0 = time.time()
    print("\n>>> %s" % " ".join(cmd), flush=True)
    with open(LOG, "a") as log:
        log.write("\n>>> %s  [%s]\n" % (" ".join(cmd), time.strftime("%Y-%m-%d %H:%M")))
        log.flush()
        p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line); log.write(line)
        rc = p.wait()
    print("<<< rc=%d  [%.1f min]" % (rc, (time.time() - t0) / 60), flush=True)
    if rc != 0:
        sys.exit("step failed: %s (see %s)" % (" ".join(cmd), LOG))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=STEPS, default=STEPS)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    py = sys.executable
    print("Aligned re-run; environment:", " ".join("%s=%s" % kv for kv in ENV.items()))
    if any(s != "summary" for s in a.only):
        preflight()
    t0 = time.time()
    if "final" in a.only:
        run([py, "run_final_model.py"])
    if "single" in a.only:
        run([py, "run_single_task.py"])
    if "ablation" in a.only:
        run([py, "run_ablation_grid.py", "--workers", str(a.workers)])
    if "huttunen" in a.only:
        run([py, "run_huttunen.py"], {"HUTTUNEN_SIGNALS": "5"})
    if "summary" in a.only:
        run([py, "summarize_aligned_controls.py"])
    print("\nall requested steps finished in %.1f h" % ((time.time() - t0) / 3600))


if __name__ == "__main__":
    main()
