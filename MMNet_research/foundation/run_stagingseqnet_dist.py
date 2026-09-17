"""Recover StagingSeqNet's predicted stage distribution, which the original run
never stored.

Section VI-N describes how the healthy-trained models fail on stroke, and the
description needs the predicted stage mix: the models do not merely lose accuracy,
they relabel most of the night Wake and stop emitting N2. run_zeroshot_seeded.py
kept pred_dist for the CNN and the CRNN, run_stagingseqnet_seeded.py did not, so
the claim could only be made for two of the three checkpoints.

Nothing here is a new experiment. It calls the original module's pretrain() with
the original seeds, so the code path and the initialisation are the ones that
produced the published figures, and writes to its own file so the published one is
never overwritten. The accuracies are compared against the stored values on the way
out: if they do not reproduce, the distributions are not to be trusted either and
the script says so rather than writing a number the paper would cite.

  KMP_DUPLICATE_LIB_OK=TRUE python run_stagingseqnet_dist.py
"""
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import run_stagingseqnet_seeded as S     # noqa: E402

FINAL = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final")
OUT = os.path.join(FINAL, "stagingseqnet_dist.json")
PUBLISHED = os.path.join(FINAL, "stagingseqnet_seeded.json")
STAGES = ["W", "N1", "N2", "N3", "R"]
TOL = 5e-4          # the published values are rounded; this is well inside that


def rich_score(model, items):
    """score() from the original module, plus the predicted mix and per-class recall."""
    yt, yp = [], []
    for x, y in items:
        yp.append(S.predict(model, x))
        yt.append(y)
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    base = S.score.__wrapped__(model, items) if hasattr(S.score, "__wrapped__") else None
    recall = []
    for k in range(5):
        m = yt == k
        recall.append(float((yp[m] == k).mean()) if m.sum() else float("nan"))
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    return dict(acc=float(accuracy_score(yt, yp)),
                mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                kappa=float(cohen_kappa_score(yt, yp)),
                n_epochs=int(len(yt)),
                pred_dist=[float(v) for v in np.bincount(yp, minlength=5) / len(yp)],
                true_dist=[float(v) for v in np.bincount(yt, minlength=5) / len(yt)],
                recall={STAGES[k]: recall[k] for k in range(5)})


def main():
    sedf = S.T.sedf_list()
    isl = S.T.isl_list()
    print("sleep-edf %d recordings | iSLEEPS %d patients" % (len(sedf), len(isl)), flush=True)
    for r in sedf:
        S.T.load_sedf(r)
    for s in isl:
        S.T.load_isleeps(s)
    sedf_k = [("s", r) for r in sedf]

    healthy_items = [(S.norm(np.load(os.path.join(S.T.SEDF, r + ".npz"))["x"].astype(np.float32)),
                      np.load(os.path.join(S.T.SEDF, r + ".npz"))["y"].astype(np.int64))
                     for r in sedf]
    stroke_items = []
    for f in sorted(glob.glob(os.path.join(S.T.PROC7, "SN*.npz")),
                    key=lambda p: int(os.path.basename(p)[2:-4])):
        sid = int(os.path.basename(f)[2:-4])
        if sid in S.T.DUPLICATE_DROP:
            continue
        d = np.load(f)
        stroke_items.append((S.norm(d["x"][:, S.T.ISL_COLS, :].astype(np.float32)),
                             d["y"].astype(np.int64)))

    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    t0 = time.time()
    for seed in S.SEEDS:
        key = "seed|%d" % seed
        if key in res:
            print("[skip]", key, flush=True)
            continue
        print("\n=== pretrain seed %d ===" % seed, flush=True)
        model = S.pretrain(seed, sedf_k)
        res[key] = dict(healthy=rich_score(model, healthy_items),
                        stroke=rich_score(model, stroke_items), seed=seed)
        json.dump(res, open(OUT, "w"), indent=1)
        print("  seed %2d  healthy %.4f  stroke %.4f  [%.1f min]"
              % (seed, res[key]["healthy"]["acc"], res[key]["stroke"]["acc"],
                 (time.time() - t0) / 60), flush=True)

    # ---- does this reproduce the published run? ----------------------------
    pub = json.load(open(PUBLISHED))
    print("\nreproduction check against %s" % os.path.basename(PUBLISHED))
    ok = True
    for seed in S.SEEDS:
        for dom in ("healthy", "stroke"):
            a = res["seed|%d" % seed][dom]["acc"]
            b = pub["seed|%d" % seed][dom]["acc"]
            good = abs(a - b) < TOL
            ok &= good
            print("  seed %2d %-8s %.4f vs %.4f  %s"
                  % (seed, dom, a, b, "ok" if good else "DIFFERS"))
    res["_reproduces_published"] = bool(ok)

    if ok:
        w = [res["seed|%d" % s]["stroke"]["pred_dist"][0] for s in S.SEEDS]
        n2 = [res["seed|%d" % s]["stroke"]["pred_dist"][2] for s in S.SEEDS]
        n2r = [res["seed|%d" % s]["stroke"]["recall"]["N2"] for s in S.SEEDS]
        res["_summary"] = dict(
            wake_share_mean=round(float(np.mean(w)), 4),
            wake_share_range=[round(float(min(w)), 4), round(float(max(w)), 4)],
            n2_share_mean=round(float(np.mean(n2)), 4),
            n2_recall_mean=round(float(np.mean(n2r)), 4))
        print("\nStagingSeqNet on stroke: Wake %.1f%% of predictions (%.1f-%.1f), "
              "N2 %.1f%%, N2 recall %.3f"
              % (100 * np.mean(w), 100 * min(w), 100 * max(w),
                 100 * np.mean(n2), np.mean(n2r)))
    else:
        print("\nthe run did not reproduce; do not cite these distributions")

    json.dump(res, open(OUT, "w"), indent=1)
    print("saved -> %s" % OUT)


if __name__ == "__main__":
    main()
