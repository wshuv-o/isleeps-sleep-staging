"""Train sleep stagers on HEALTHY sleep only, then measure them on iSLEEPS.

The existing evidence for the domain-gap claim rests on one checkpoint. If that
checkpoint happens to be fragile, the claim is about the checkpoint rather than
about the cohort. So two further architectures are trained here from random
initialisation on Sleep-EDF alone -- no iSLEEPS data touches training -- and each
is scored twice: on held-out HEALTHY recordings (the control) and on all 99
iSLEEPS patients (the transfer test).

A model is only evidence of a domain gap if it works in its own domain. Reporting
both numbers per model is what separates "the injured brain breaks stagers" from
"these stagers were never any good".

Architectures are deliberately conventional, not tuned, and independent of
anything in this repository:
  cnn   per-epoch convolutional stager
  crnn  the same encoder with a BiLSTM over 20-epoch sequences

Channel set is the one that pairs across corpora, per train_transfer.py:
  Sleep-EDF  Fpz-Cz, Pz-Oz, EOG horizontal, EMG submental
  iSLEEPS    C4:M1,  O2:M1, E1:M2,          EMG   (processed7 columns 0,2,4,6)
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from sedf_control import load_recording                      # noqa: E402

RAW = os.path.join(REPO, "data", "sleep_edf_raw")
CACHE = os.path.join(REPO, "data", "sleep_edf_proc")
P7 = os.path.join(REPO, "data", "processed7")
OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "domaingap")
ISL_COLS = [0, 2, 4, 6]
DUP = {28}
L, NCLS = 20, 5
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STAGES = ["W", "N1", "N2", "N3", "R"]


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def build_cache():
    os.makedirs(CACHE, exist_ok=True)
    psgs = sorted(glob.glob(os.path.join(RAW, "*-PSG.edf")))
    hyps = {os.path.basename(p)[:6]: p for p in glob.glob(os.path.join(RAW, "*-Hypnogram.edf"))}
    made = 0
    for p in psgs:
        stem = os.path.basename(p)[:6]
        dst = os.path.join(CACHE, stem + ".npz")
        if stem not in hyps or os.path.exists(dst):
            continue
        x, y, err = load_recording(p, hyps[stem])
        if err:
            print("  skip %s: %s" % (stem, err)); continue
        np.savez_compressed(dst, x=x.astype(np.float16), y=y)
        made += 1
        print("  cached %s  %5d epochs" % (stem, len(y)), flush=True)
    print("cache: %d new, %d total" % (made, len(glob.glob(os.path.join(CACHE, "*.npz")))))


def _norm(x):
    """Per-recording z-score over (epoch, time) per channel."""
    mu = x.mean((0, 2), keepdims=True)
    sd = x.std((0, 2), keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


def load_sedf():
    out = {}
    for f in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        d = np.load(f)
        out[os.path.basename(f)[:-4]] = (_norm(d["x"].astype(np.float32)), d["y"])
    return out


def load_isleeps():
    out = {}
    for f in sorted(glob.glob(os.path.join(P7, "SN*.npz")),
                    key=lambda p: int(os.path.basename(p)[2:-4])):
        sid = int(os.path.basename(f)[2:-4])
        if sid in DUP:
            continue
        d = np.load(f)
        out[sid] = (_norm(d["x"][:, ISL_COLS, :].astype(np.float32)), d["y"].astype(np.int64))
    return out


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------
class Encoder(nn.Module):
    """Conventional two-scale convolutional epoch encoder -> 128-d."""

    def __init__(self, in_ch=4, drop=0.5):
        super().__init__()
        def branch(k, s, p):
            return nn.Sequential(
                nn.Conv1d(in_ch, 64, k, s, p), nn.BatchNorm1d(64), nn.GELU(),
                nn.MaxPool1d(8, 8), nn.Dropout(drop),
                nn.Conv1d(64, 128, 8, 1, 4), nn.BatchNorm1d(128), nn.GELU(),
                nn.Conv1d(128, 128, 8, 1, 4), nn.BatchNorm1d(128), nn.GELU(),
                nn.AdaptiveAvgPool1d(1))
        self.fine = branch(50, 6, 24)        # ~0.5 s kernel: transients, spindles
        self.coarse = branch(200, 16, 100)   # ~2 s kernel: slow waves
        self.proj = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(drop))

    def forward(self, x):
        f = torch.cat([self.fine(x).flatten(1), self.coarse(x).flatten(1)], 1)
        return self.proj(f)


class CNN(nn.Module):
    def __init__(self, in_ch=4, drop=0.5):
        super().__init__()
        self.enc = Encoder(in_ch, drop)
        self.head = nn.Linear(128, NCLS)

    def forward(self, x):                     # x [B, L, C, T]
        B, Ln, C, T = x.shape
        f = self.enc(x.reshape(B * Ln, C, T))
        return self.head(f).reshape(B, Ln, NCLS)


class CRNN(nn.Module):
    def __init__(self, in_ch=4, drop=0.5, hidden=128):
        super().__init__()
        self.enc = Encoder(in_ch, drop)
        self.lstm = nn.LSTM(128, hidden, 2, batch_first=True,
                            bidirectional=True, dropout=drop)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * hidden, NCLS))

    def forward(self, x):
        B, Ln, C, T = x.shape
        f = self.enc(x.reshape(B * Ln, C, T)).reshape(B, Ln, 128)
        return self.head(self.lstm(f)[0])


MODELS = {"cnn": CNN, "crnn": CRNN}


# ----------------------------------------------------------------------------
# train / evaluate
# ----------------------------------------------------------------------------
def windows(store, keys, stride):
    idx = []
    for k in keys:
        n = len(store[k][1])
        if n < L:
            continue
        for s in range(0, n - L + 1, stride):
            idx.append((k, s))
    return idx


def batch(store, items):
    xs = np.stack([store[k][0][s:s + L] for k, s in items])
    ys = np.stack([store[k][1][s:s + L] for k, s in items])
    return (torch.tensor(xs, device=DEV), torch.tensor(ys, dtype=torch.long, device=DEV))


@torch.no_grad()
def evaluate(model, store, keys):
    model.eval()
    yt, yp = [], []
    for k in keys:
        x, y = store[k]
        n = len(y)
        pad = (-n) % L
        xx = np.concatenate([x, np.zeros((pad,) + x.shape[1:], np.float32)]) if pad else x
        seq = xx.reshape(-1, L, x.shape[1], x.shape[2])
        out = []
        for i in range(0, len(seq), 32):
            b = torch.tensor(seq[i:i + 32], device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(model(b).float().cpu().numpy())
        pred = np.concatenate(out).reshape(-1, NCLS)[:n].argmax(1)
        yt.append(y); yp.append(pred)
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return dict(acc=float(accuracy_score(yt, yp)),
                mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                kappa=float(cohen_kappa_score(yt, yp)),
                n_epochs=int(len(yt)),
                pred_dist=[float(v) for v in np.bincount(yp, minlength=5) / len(yp)],
                true_dist=[float(v) for v in np.bincount(yt, minlength=5) / len(yt)])


def train(name, sedf, tr_keys, va_keys, epochs, seed, bs=32, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MODELS[name](in_ch=4).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    cnt = np.zeros(5, np.int64)
    for k in tr_keys:
        cnt += np.bincount(sedf[k][1], minlength=5)
    w = np.sqrt(cnt.sum() / (5 * np.maximum(cnt, 1))); w = w / w.mean()
    ce = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEV))
    idx = windows(sedf, tr_keys, L // 2)
    rng = np.random.RandomState(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(idx))
        tot = 0.0
        for i in range(0, len(order) - bs + 1, bs):
            xb, yb = batch(sedf, [idx[j] for j in order[i:i + bs]])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = ce(model(xb).reshape(-1, NCLS).float(), yb.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += float(loss)
        sch.step()
        va = evaluate(model, sedf, va_keys)
        print("    epoch %2d  loss %.4f  val acc %.4f" % (ep, tot / max(1, len(order) // bs), va["acc"]),
              flush=True)
        if va["acc"] > best:
            best, bad = va["acc"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    model.load_state_dict(best_state)
    return model, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["cnn", "crnn"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("building Sleep-EDF cache ...", flush=True)
    build_cache()
    sedf = load_sedf()
    keys = sorted(sedf)
    rng = np.random.RandomState(0); order = rng.permutation(len(keys))
    va_keys = [keys[i] for i in order[:8]]
    tr_keys = [keys[i] for i in order[8:]]
    print("Sleep-EDF: %d train / %d val recordings" % (len(tr_keys), len(va_keys)), flush=True)

    isl = load_isleeps()
    print("iSLEEPS  : %d subjects" % len(isl), flush=True)

    results = {}
    path = os.path.join(OUT, "healthy_trained_models.json")
    if os.path.exists(path):
        results = json.load(open(path))
    for name in a.models:
        if name in results:
            print("[skip]", name); continue
        t0 = time.time()
        print("\n=== training %s on HEALTHY Sleep-EDF only ===" % name, flush=True)
        model, best = train(name, sedf, tr_keys, va_keys, a.epochs, a.seed)
        healthy = evaluate(model, sedf, va_keys)
        stroke = evaluate(model, isl, sorted(isl))
        results[name] = dict(healthy=healthy, stroke=stroke,
                             params=int(sum(p.numel() for p in model.parameters())),
                             minutes=(time.time() - t0) / 60)
        json.dump(results, open(path, "w"), indent=1)
        print("  %-5s healthy acc %.4f kappa %.4f   |   stroke acc %.4f kappa %.4f   [%.1f min]"
              % (name, healthy["acc"], healthy["kappa"], stroke["acc"], stroke["kappa"],
                 results[name]["minutes"]), flush=True)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
