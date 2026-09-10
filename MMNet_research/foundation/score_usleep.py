"""Score a U-Sleep hypnogram against the iSLEEPS reference scoring.

U-Sleep (Perslev et al., npj Digital Medicine 2021) runs on an external server,
so its predictions arrive as downloaded hypnogram files rather than from a
training run of ours. This turns one or many of those files into the same
accuracy / macro-F1 / kappa the rest of the table reports.

Accepts whatever the web interface hands back -- a one-stage-per-line text file,
or the ``.tsv``/``.npy`` forms the API returns -- and matches it against the
labels in data/processed7. U-Sleep predicts on its own grid; if the returned
hypnogram is longer or shorter than our epoch count the overlap is scored and
the mismatch is printed, because silently truncating a misaligned hypnogram is
how a benchmark row becomes wrong.

  python score_usleep.py hypnograms/*.tsv
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
P7 = os.path.join(REPO, "data", "processed7")
OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final",
                   "usleep.json")
STAGES = ["W", "N1", "N2", "N3", "R"]
MODEL = "v2.0"

# U-Sleep emits several naming conventions depending on version and export path
MAP = {"W": 0, "WAKE": 0, "AWAKE": 0, "0": 0,
       "N1": 1, "S1": 1, "1": 1,
       "N2": 2, "S2": 2, "2": 2,
       "N3": 3, "S3": 3, "S4": 3, "N4": 3, "3": 3,
       "R": 4, "REM": 4, "4": 4, "5": 4}


def read_hypnogram(path):
    if path.endswith(".npy"):
        a = np.load(path)
        # the web interface saves the softmax, one row of 5 class scores per
        # epoch, rather than the argmax hypnogram the API returns
        if a.ndim == 2 and a.shape[1] == 5:
            return a.argmax(1).astype(np.int64)
        raw = [str(v) for v in a.ravel()]
    else:
        raw = []
        for line in open(path):
            line = line.strip()
            if not line or line.lower().startswith(("stage", "#", "init")):
                continue
            # tsv exports are "start<TAB>duration<TAB>stage"; take the last field
            raw.append(line.replace(",", "\t").split("\t")[-1].strip())
    out = []
    for v in raw:
        k = v.upper().strip().strip('"')
        if k in MAP:
            out.append(MAP[k])
        elif k in ("?", "A", "ARTIFACT", "UNKNOWN", "MOVEMENT", "M"):
            out.append(-1)                       # excluded from scoring
        else:
            raise SystemExit("unrecognised stage label %r in %s" % (v, path))
    return np.array(out, dtype=np.int64)


def reference(sid):
    f = os.path.join(P7, "SN%d.npz" % sid)
    if not os.path.exists(f):
        raise SystemExit("no reference scoring for SN%d" % sid)
    return np.load(f, allow_pickle=True)["y"].astype(np.int64)


def sid_of(path):
    base = os.path.basename(path)
    digits = ""
    for ch in base[base.upper().find("SN") + 2:]:
        if ch.isdigit():
            digits += ch
        else:
            break
    if not digits:
        raise SystemExit("cannot read a recording id out of %r" % base)
    return int(digits)


def main(paths):
    yt_all, yp_all, per_rec = [], [], []
    for p in sorted(paths):
        sid = sid_of(p)
        yp, yt = read_hypnogram(p), reference(sid)
        if len(yp) != len(yt):
            print("  SN%-4d length mismatch: predicted %d, reference %d -- "
                  "scoring the leading %d" % (sid, len(yp), len(yt),
                                              min(len(yp), len(yt))))
        k = min(len(yp), len(yt))
        yp, yt = yp[:k], yt[:k]
        keep = yp >= 0
        yp, yt = yp[keep], yt[keep]
        per_rec.append(dict(recording="SN%d" % sid, n=int(len(yt)),
                            acc=float(accuracy_score(yt, yp)),
                            mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                            kappa=float(cohen_kappa_score(yt, yp))))
        print("SN%-4d %5d ep  acc %.4f  mF1 %.4f  kappa %.4f"
              % (sid, len(yt), per_rec[-1]["acc"], per_rec[-1]["mf1"],
                 per_rec[-1]["kappa"]))
        yt_all.append(yt); yp_all.append(yp)

    yt, yp = np.concatenate(yt_all), np.concatenate(yp_all)
    pooled = dict(acc=float(accuracy_score(yt, yp)),
                  mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                  kappa=float(cohen_kappa_score(yt, yp)))
    # across-recording dispersion, so the row can carry an interval like the rest
    disp = {k: float(np.std([r[k] for r in per_rec], ddof=1)) if len(per_rec) > 1
            else None for k in ("acc", "mf1", "kappa")}
    pcf = f1_score(yt, yp, average=None, labels=range(5), zero_division=0)

    res = dict(model="U-Sleep %s (zero-shot)" % MODEL,
               reference="Perslev et al., npj Digital Medicine 2021",
               protocol="pretrained checkpoint applied cold, no training on iSLEEPS",
               n_recordings=len(per_rec), n_epochs=int(len(yt)),
               pooled=pooled, sd_across_recordings=disp, per_recording=per_rec,
               per_class_f1={STAGES[i]: float(pcf[i]) for i in range(5)})
    json.dump(res, open(OUT, "w"), indent=1)

    print("\nU-SLEEP, %d recordings, %d epochs" % (len(per_rec), len(yt)))
    for k in ("acc", "mf1", "kappa"):
        s = "" if disp[k] is None else "  (SD across recordings %.4f)" % disp[k]
        print("  %-6s %.4f%s" % (k, pooled[k], s))
    print("  per-class F1:", {STAGES[i]: round(float(pcf[i]), 3) for i in range(5)})
    print("\n  confusion (rows = reference, cols = U-Sleep), %s" % " ".join(STAGES))
    for i, row in enumerate(confusion_matrix(yt, yp, labels=range(5))):
        print("   %-3s %s" % (STAGES[i], " ".join("%6d" % v for v in row)))
    print("\nwrote", OUT)


if __name__ == "__main__":
    args = sys.argv[1:]
    # which checkpoint produced these files -- the web interface offers v1.0,
    # v2.0, the EEG-only variant and a fine-tuned one, and they are not the same
    # model, so the row has to say which it is
    MODEL = "v2.0"
    for a in list(args):
        if a.startswith("--model="):
            MODEL = a.split("=", 1)[1]; args.remove(a)
    files = [f for a in args for f in (glob.glob(a) if any(c in a for c in "*?") else [a])]
    if not files:
        raise SystemExit(__doc__)
    main(files)
