"""Event-level respiratory detection and end-to-end AHI, on the paper's ten folds.

Input: the eight cardiorespiratory channels at 25 Hz (ECG, thermal flow, thorax, abdomen, effort,
SpO2, pulse from data/multimodal; nasal pressure from data/pressure_flow), z-scored per patient.
Target: 1-second apnea/hypopnea mask from the annotation workbooks (data/resp_events).

Detector: a 1-D convolutional encoder (25 Hz -> 1 Hz) and a U-Net over 10-minute windows with a
bidirectional GRU at the coarsest scale, giving an event probability every second.

Events: per-second probabilities thresholded, runs shorter than 10 s dropped, gaps under 3 s
merged; the threshold is chosen on the fold's validation patients (event F1).
AHI: detected events starting in predicted sleep / predicted total sleep time, where sleep comes
from the final MM-Net's staging (3-seed probabilities, train-fold HMM) for the same patients.

Reported on test patients, against the annotation AHI (primary) and clinical AHI_1_B:
per-second AUC, event-level F1 (IoU >= 0.2), Spearman rho, MAE, severity class accuracy and
linearly weighted kappa (AASM cuts 5 / 15 / 30), plus an oracle row using the true hypnogram.

    python run_event_ahi.py [--seed 42] [--stages base-nocw-cross-ftlae-p8]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score

import mmnet_core as C

DATA = os.path.join(REPO, "data")
RUNS = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "lesion")
ENS = os.path.join(RUNS, "seed_ensemble")
OUT = os.path.join(RUNS, "event_ahi")
DEV = "cuda"
FS, WIN_S = 25, 600
CUTS = [5, 15, 30]


def split(fi, tr_all):
    rng = np.random.RandomState(100 + fi)
    tr_all = list(tr_all)
    rng.shuffle(tr_all)
    nv = max(10, len(tr_all) // 9)
    return tr_all[nv:], tr_all[:nv]


ALIGNED, USE_EEG = False, False      # --aligned: data/aligned (EEG grid); --eeg: + processed7 EEG/EOG/EMG at 100 Hz


def load_patient(s):
    base = os.path.join(DATA, "aligned") if ALIGNED else DATA
    d = np.load(os.path.join(base, "multimodal", "SN%d.npz" % s))
    card = d["card"].astype(np.float32)                       # [n, 7, 750]
    n = card.shape[0]
    p = np.load(os.path.join(base, "pressure_flow", "SN%d.npz" % s))
    pf = p["pflow"].astype(np.float32)[:n][:, None, :]         # [n, 1, 750]
    if not bool(p["valid"]):
        pf[:] = 0
    x = np.concatenate([card, pf], 1).transpose(1, 0, 2).reshape(8, n * 750)
    m, sd = x.mean(1, keepdims=True), x.std(1, keepdims=True)
    x = np.where(sd > 1e-6, (x - m) / (sd + 1e-6), 0.0).astype(np.float16)
    y = np.load(os.path.join(base, "resp_events", "SN%d.npz" % s))["mask"][: n * 30].astype(np.float32)
    if USE_EEG:
        e = np.load(os.path.join(DATA, "processed7", "SN%d.npz" % s))["x"][:n].astype(np.float32)   # [n, 7, 3000] @100 Hz
        e = e.transpose(1, 0, 2).reshape(7, n * 3000)
        m2, sd2 = e.mean(1, keepdims=True), e.std(1, keepdims=True)
        e = np.where(sd2 > 1e-6, (e - m2) / (sd2 + 1e-6), 0.0).astype(np.float16)
        return (x, e), y
    return x, y


class Block(nn.Module):
    def __init__(self, i, o, k):
        super().__init__()
        self.c = nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.GELU(),
                               nn.Conv1d(o, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.GELU())

    def forward(self, x):
        return self.c(x)


class EventNet(nn.Module):
    """[b, 8, 25*T] -> [b, T] logits (one per second)."""

    def __init__(self, w=64, eeg=False):
        super().__init__()
        self.eeg = nn.Sequential(Block(7, 16, 15), nn.MaxPool1d(4)) if eeg else None     # 100 -> 25 Hz
        self.front = nn.Sequential(Block(8 + (16 if eeg else 0), 32, 11), nn.MaxPool1d(5), Block(32, w, 7), nn.MaxPool1d(5))
        self.d1, self.d2, self.d3 = Block(w, w, 5), Block(w, 2 * w, 5), Block(2 * w, 2 * w, 5)
        self.gru = nn.GRU(2 * w, w, batch_first=True, bidirectional=True)
        self.u3, self.u2, self.u1 = Block(4 * w, 2 * w, 5), Block(4 * w, w, 5), Block(2 * w, w, 5)
        self.head = nn.Conv1d(w, 1, 1)

    def forward(self, x, e=None):
        if self.eeg is not None:
            x = torch.cat([x, self.eeg(e)[..., :x.shape[-1]]], 1)
        f = self.front(x)                          # 1 Hz
        a = self.d1(f)                             # T
        b = self.d2(F.max_pool1d(a, 2))            # T/2
        c = self.d3(F.max_pool1d(b, 2))            # T/4
        g, _ = self.gru(F.max_pool1d(c, 2).transpose(1, 2))   # T/8
        g = F.interpolate(g.transpose(1, 2), size=c.shape[-1])
        c = self.u3(torch.cat([c, g], 1))
        b = self.u2(torch.cat([b, F.interpolate(c, size=b.shape[-1])], 1))
        a = self.u1(torch.cat([a, F.interpolate(b, size=a.shape[-1])], 1))
        return self.head(a).squeeze(1)


@torch.no_grad()
def predict(model, x):
    """Overlapping 10-minute windows (stride 5 min), averaged -> per-second probability."""
    model.eval()
    x, e = x if isinstance(x, tuple) else (x, None)
    T = x.shape[1] // FS
    out, cnt = np.zeros(T, np.float32), np.zeros(T, np.float32)
    starts = list(range(0, max(1, T - WIN_S + 1), WIN_S // 2))
    if starts[-1] + WIN_S < T:
        starts.append(max(0, T - WIN_S))
    for st in starts:
        seg = x[:, st * FS:(st + WIN_S) * FS]
        L = seg.shape[1] // FS
        if L < WIN_S:
            seg = np.pad(seg, ((0, 0), (0, (WIN_S - L) * FS)))
        eb = None
        if e is not None:
            eseg = e[:, st * 100:(st + WIN_S) * 100]
            if eseg.shape[1] < WIN_S * 100:
                eseg = np.pad(eseg, ((0, 0), (0, WIN_S * 100 - eseg.shape[1])))
            eb = torch.from_numpy(eseg[None].astype(np.float32)).to(DEV)
        p = torch.sigmoid(model(torch.from_numpy(seg[None].astype(np.float32)).to(DEV), eb))[0, :L]
        out[st:st + L] += p.float().cpu().numpy(); cnt[st:st + L] += 1
    return out / np.maximum(cnt, 1)


def train(X, Y, tr, va, seed, epochs=25, bs=24, patience=5):
    torch.manual_seed(seed); np.random.seed(seed)
    model = EventNet(eeg=USE_EEG).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = np.mean(np.concatenate([Y[s] for s in tr]))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.sqrt((1 - pos) / max(pos, 1e-4)), device=DEV))
    starts = [(s, st) for s in tr for st in range(0, max(1, len(Y[s]) - WIN_S), WIN_S // 4)]
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(len(starts))
        for i in range(0, len(order) - bs + 1, bs):
            xb, eb, yb = [], [], []
            for j in order[i:i + bs]:
                s, st = starts[j]
                st = min(st + np.random.randint(0, WIN_S // 4), max(0, len(Y[s]) - WIN_S))
                xs, es = X[s] if USE_EEG else (X[s], None)
                xb.append(xs[:, st * FS:(st + WIN_S) * FS]); yb.append(Y[s][st:st + WIN_S])
                if es is not None:
                    eb.append(es[:, st * 100:(st + WIN_S) * 100])
            xb = torch.from_numpy(np.stack(xb).astype(np.float32)).to(DEV)
            eb = torch.from_numpy(np.stack(eb).astype(np.float32)).to(DEV) if USE_EEG else None
            yb = torch.from_numpy(np.stack(yb)).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                lo = model(xb, eb)
            loss = lossf(lo.float(), yb)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        yv = np.concatenate([Y[s] for s in va]); pv = np.concatenate([predict(model, X[s]) for s in va])
        auc = roc_auc_score(yv, pv) if 0 < yv.mean() < 1 else 0.5
        if auc > best:
            best, bad, best_state = auc, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best


def segments(binary):
    d = np.diff(np.concatenate([[0], binary.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def events_from(p, thr, min_len=10, gap=3):
    seg = segments(p >= thr)
    merged = []
    for a, b in seg:
        if merged and a - merged[-1][1] < gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if b - a >= min_len]


def event_f1(pred, true, iou=0.2):
    used, tp = set(), 0
    for a, b in pred:
        for k, (c, d) in enumerate(true):
            if k in used:
                continue
            inter = max(0, min(b, d) - max(a, c))
            if inter > 0 and inter / (max(b, d) - min(a, c)) >= iou:
                used.add(k); tp += 1
                break
    prec = tp / max(1, len(pred)); rec = tp / max(1, len(true))
    return 2 * prec * rec / max(1e-9, prec + rec), tp, len(pred), len(true)


def ahi_of(events, stages):
    """Events starting in a sleep epoch per hour of sleep, on a stage sequence (0 = Wake)."""
    sleep = stages != 0
    tst = sleep.sum() * 30 / 3600
    n = sum(1 for a, _ in events if a // 30 < len(stages) and sleep[int(a // 30)])
    return n / tst if tst > 0 else 0.0


def metrics(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    ct, cp = np.digitize(true, CUTS), np.digitize(pred, CUTS)
    return dict(rho=float(spearmanr(pred, true).correlation), mae=float(np.abs(pred - true).mean()),
                sev_acc=float(accuracy_score(ct, cp)), sev_kappa=float(cohen_kappa_score(ct, cp, weights="linear")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stages", default="base-nocw-cross-ftlae-p8")
    ap.add_argument("--aligned", action="store_true")
    ap.add_argument("--eeg", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    global ALIGNED, USE_EEG, OUT
    ALIGNED, USE_EEG = a.aligned, a.eeg
    if a.tag:
        OUT = os.path.join(RUNS, "event_ahi_" + a.tag)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    X, Y = {}, {}
    for s in C.SUBS:
        X[s], Y[s] = load_patient(s)
    if ALIGNED:      # events as runs of the EEG-grid mask, so events and stages share one timeline
        true_ev = {s: [(a_, b_) for a_, b_ in segments(Y[s] > 0)] for s in C.SUBS}
        ann_ahi = {s: float(np.load(os.path.join(DATA, "aligned", "resp_events", "SN%d.npz" % s))["ahi_ann"]) for s in C.SUBS}
    else:
        true_ev = {s: [(e[0], e[1]) for e in np.load(os.path.join(DATA, "resp_events", "SN%d.npz" % s))["events"]
                       if 1 <= e[2] <= 4] for s in C.SUBS}
        ann_ahi = {s: float(np.load(os.path.join(DATA, "resp_events", "SN%d.npz" % s))["ahi_ann"]) for s in C.SUBS}
    col = pd.to_numeric(pd.read_excel(os.path.join(DATA, "full100", "subject_description.xlsx"))["AHI_1_B"], errors="coerce")
    clin = {i + 1: float(v) for i, v in enumerate(col) if not pd.isna(v)}
    print("loaded %d patients in %.1f min" % (len(X), (time.time() - t0) / 60), flush=True)

    rows, f1s, aucs = {}, [], []
    for fi, (tr_all, te) in enumerate(C.FOLDS):
        path = os.path.join(OUT, "fold%d_s%d.json" % (fi, a.seed))
        if os.path.exists(path):
            r = json.load(open(path))
        else:
            tr, va = split(fi, tr_all)
            model, vauc = train(X, Y, tr, va, a.seed)
            P = {s: predict(model, X[s]) for s in list(va) + list(te)}
            np.savez_compressed(os.path.join(OUT, "fold%d_s%d_prob.npz" % (fi, a.seed)),
                                **{"SN%d" % s_: P[s_].astype(np.float16) for s_ in P})
            # threshold chosen on validation patients so that the AHI comes out right (median
            # absolute AHI error, true hypnogram as the denominator); event F1 is reported too
            best_thr, best_err, best_f1 = 0.5, np.inf, 0.0
            for thr in np.arange(0.1, 0.85, 0.025):
                errs, tp = [], 0
                npred = ntrue = 0
                for s in va:
                    ev = events_from(P[s], thr)
                    errs.append(abs(ahi_of(ev, C.DATA[s][2]) - ann_ahi[s]))
                    _, t_, p_, n_ = event_f1(ev, true_ev[s])
                    tp += t_; npred += p_; ntrue += n_
                err = float(np.median(errs))
                if err < best_err:
                    best_err, best_thr, best_f1 = err, float(thr), 2 * tp / max(1, npred + ntrue)
            st = np.load(os.path.join(ENS, "%s_fold%d.npz" % (a.stages, fi)))
            r = dict(fold=fi, val_auc=vauc, thr=best_thr, val_f1=best_f1, val_ahi_err=best_err, patients={})
            for s in te:
                stg = C.hmm(st["A_log"], st["pi_log"],
                            np.log(np.mean([st["SN%d|%d" % (s, sd)] for sd in (42, 1, 7)], 0) + C.EPS))
                ev = events_from(P[s], best_thr)
                f, tp, npred, ntrue = event_f1(ev, true_ev[s])
                r["patients"][str(s)] = dict(ahi_pred=ahi_of(ev, stg), ahi_pred_truestage=ahi_of(ev, C.DATA[s][2]),
                                             tp=tp, npred=npred, ntrue=ntrue,
                                             auc=float(roc_auc_score(Y[s], P[s])) if 0 < Y[s].mean() < 1 else None)
            json.dump(r, open(path, "w"), indent=1)
            print("fold %d: val AUC %.3f, thr %.3f, val median AHI error %.1f, val event F1 %.3f [%.1f min]"
                  % (fi, vauc, best_thr, best_err, best_f1, (time.time() - t0) / 60), flush=True)
        rows.update({int(k): v for k, v in r["patients"].items()})

    subs = sorted(rows)
    tp = sum(rows[s]["tp"] for s in subs); npred = sum(rows[s]["npred"] for s in subs); ntrue = sum(rows[s]["ntrue"] for s in subs)
    lines = ["Event-level detector, seed %d, 10 test folds, %d patients" % (a.seed, len(subs)),
             "per-second AUC (mean over patients) %.3f | event F1 %.3f (precision %.3f, recall %.3f)"
             % (np.nanmean([rows[s]["auc"] for s in subs if rows[s]["auc"] is not None]),
                2 * tp / max(1, npred + ntrue), tp / max(1, npred), tp / max(1, ntrue)), ""]
    for lab, key in (("AHI, predicted stages (end to end)", "ahi_pred"), ("AHI, true stages (oracle)", "ahi_pred_truestage")):
        p = [rows[s][key] for s in subs]
        m1 = metrics(p, [ann_ahi[s] for s in subs])
        cs = [s for s in subs if s in clin]
        m2 = metrics([rows[s][key] for s in cs], [clin[s] for s in cs])
        lines.append("%-36s vs annotation AHI: rho %.3f MAE %.1f severity acc %.2f kappa %.2f | vs clinical: rho %.3f MAE %.1f acc %.2f kappa %.2f"
                     % (lab, m1["rho"], m1["mae"], m1["sev_acc"], m1["sev_kappa"], m2["rho"], m2["mae"], m2["sev_acc"], m2["sev_kappa"]))
    text = "\n".join(lines)
    open(os.path.join(RUNS, "event_ahi_s%d.txt" % a.seed), "w").write(text + "\n")
    print("\n" + text, flush=True)


if __name__ == "__main__":
    main()
