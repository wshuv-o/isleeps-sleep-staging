"""StagingSeqNet pretrained three times, so its zero-shot row can carry an interval.

The benchmark table gives this row from a single supplied checkpoint
(HAGNet_research/model/pretrained_sedf.pt), so it has no dispersion. The training
code lives in HAGNet_research/train/baselines/train_transfer.py and reads two
directories that do not exist under that tree, which is why I first reported this
as blocked. It is not: both datasets already exist at the repository root in
exactly the layout the script wants, and nothing needs preprocessing.

  PROC7  wants 7-channel SN*.npz     -> data/processed7
  SEDF   wants 4-channel x plus y    -> data/sleep_edf_proc, shape (n, 4, 3000)

So the module is imported with those two paths overridden, and only its pretrain
phase is used. The fine-tune phase that follows it in that script trains on
iSLEEPS, which is the opposite of what a zero-shot row measures.

Each seed produces a checkpoint, and each checkpoint is scored twice: on the
Sleep-EDF recordings it was trained on, which is the published healthy figure and
is in-sample, and on all 99 iSLEEPS patients, which is the transfer test.

  KMP_DUPLICATE_LIB_OK=TRUE python run_stagingseqnet_seeded.py
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TT = os.path.join(REPO, "HAGNet_research", "train", "baselines")
sys.path.insert(0, TT)
sys.path.insert(0, os.path.join(REPO, "HAGNet_research", "train", "legacy_models"))
sys.path.insert(0, os.path.join(REPO, "HAGNet_research", "utils"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import train_transfer as T          # noqa: E402
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score)  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

# the two directories that tree expects, which live at the repository root
T.PROC7 = os.path.join(REPO, "data", "processed7")
T.SEDF = os.path.join(REPO, "data", "sleep_edf_proc")

OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final",
                   "stagingseqnet_seeded.json")
SEEDS = [42, 1, 7]
PRE_EPOCHS, BATCH, LR = 15, 64, 1e-3
DEV = "cuda" if torch.cuda.is_available() else "cpu"
L = T.L


def pretrain(seed, sedf_k):
    """The pretrain phase of train_transfer, nothing after it."""
    T.set_seed(seed)
    model = T.StagingSeqNet(in_ch=4, dropout=0.5).to(DEV)
    dl = DataLoader(T.LazySeq(sedf_k, L // 2), batch_size=BATCH, shuffle=True,
                    num_workers=0, pin_memory=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PRE_EPOCHS)
    crit = nn.CrossEntropyLoss(weight=T.cw(sedf_k, DEV), reduction="none")
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))
    for ep in range(1, PRE_EPOCHS + 1):
        model.train(); tot = nb = 0
        for x, y, m in dl:
            x, y, m = x.to(DEV), y.to(DEV), m.to(DEV)
            x = T.augment(x); opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEV == "cuda")):
                lo = model(x)
                loss = crit(lo.reshape(-1, 5), y.reshape(-1))
                loss = (loss * m.reshape(-1)).sum() / m.sum().clamp(min=1)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
        sch.step()
        print("    ep %2d  loss %.4f" % (ep, tot / max(nb, 1)), flush=True)
    model.eval()
    return model


@torch.no_grad()
def predict(model, x):
    n = len(x)
    pad = (-n) % L
    if pad:
        x = np.concatenate([x, np.zeros((pad, x.shape[1], x.shape[2]), np.float32)])
    seq = x.reshape(-1, L, x.shape[1], x.shape[2])
    out = []
    for i in range(0, len(seq), 16):
        b = torch.tensor(seq[i:i + 16], dtype=torch.float32, device=DEV)
        out.append(model(b).float().cpu().numpy())
    return np.concatenate(out).reshape(-1, 5)[:n].argmax(1)


def norm(x):
    mu = x.mean((0, 2), keepdims=True)
    sd = x.std((0, 2), keepdims=True) + 1e-6
    return (x - mu) / sd


def score(model, items):
    yt, yp = [], []
    for x, y in items:
        yp.append(predict(model, x)); yt.append(y)
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return dict(acc=float(accuracy_score(yt, yp)),
                mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                kappa=float(cohen_kappa_score(yt, yp)), n_epochs=int(len(yt)))


def main():
    sedf = T.sedf_list(); isl = T.isl_list()
    print("sleep-edf %d recordings | iSLEEPS %d patients" % (len(sedf), len(isl)),
          flush=True)
    for r in sedf:
        T.load_sedf(r)
    for s in isl:
        T.load_isleeps(s)
    sedf_k = [("s", r) for r in sedf]

    healthy_items = [(norm(np.load(os.path.join(T.SEDF, r + ".npz"))["x"].astype(np.float32)),
                      np.load(os.path.join(T.SEDF, r + ".npz"))["y"].astype(np.int64))
                     for r in sedf]
    stroke_items = []
    for f in sorted(glob.glob(os.path.join(T.PROC7, "SN*.npz")),
                    key=lambda p: int(os.path.basename(p)[2:-4])):
        sid = int(os.path.basename(f)[2:-4])
        if sid in T.DUPLICATE_DROP:
            continue
        d = np.load(f)
        stroke_items.append((norm(d["x"][:, T.ISL_COLS, :].astype(np.float32)),
                             d["y"].astype(np.int64)))

    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    t0 = time.time()
    for seed in SEEDS:
        key = "seed|%d" % seed
        if key in res:
            print("[skip]", key, flush=True); continue
        print("\n=== pretrain seed %d ===" % seed, flush=True)
        model = pretrain(seed, sedf_k)
        res[key] = dict(healthy=score(model, healthy_items),
                        stroke=score(model, stroke_items), seed=seed)
        json.dump(res, open(OUT, "w"), indent=1)
        print("  seed %2d  healthy acc %.4f   stroke acc %.4f  [%.1f min]"
              % (seed, res[key]["healthy"]["acc"], res[key]["stroke"]["acc"],
                 (time.time() - t0) / 60), flush=True)

    print("\nSTAGINGSEQNET over %d seeds" % len(SEEDS))
    summary = {}
    for dom in ("healthy", "stroke"):
        for m in ("acc", "mf1", "kappa"):
            v = np.array([res["seed|%d" % s][dom][m] for s in SEEDS])
            summary["%s|%s" % (dom, m)] = [float(v.mean()), float(v.std(ddof=1))]
            print("  %-8s %-6s %.4f +- %.4f" % (dom, m, v.mean(), v.std(ddof=1)))
    res["_summary"] = summary
    res["_note"] = ("dispersion across pretraining seeds; the healthy figure is "
                    "in-sample, scored on the recordings the model pretrained on")
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
