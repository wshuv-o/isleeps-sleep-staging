"""Huttunen et al. (IEEE TBME 2023) on iSLEEPS, on the same folds as everything else.

The closest prior system to this paper: one network scoring sleep stages and
respiratory events simultaneously. Reviewers asked for the comparison, and without
it the paper argues novelty against a description instead of against a model.

The architecture is ported from the authors' own release,
github.com/rikuhuttunen/psg-simultscoring-models, not reconstructed from the
paper. It is a U-Time variant, so it shares a lineage with the U-Time row already
in Table 4, but differs in three ways that matter: squeeze-and-excitation inside
every convolutional block, atrous spatial pyramid pooling at the bottleneck, and
two segment classifiers over one shared decoder. Their DEFAULT_BLOCK_ARGS are used
verbatim:

    kernel 5, filters  32, pool 8, dilation 2, se 0.25
    kernel 5, filters  48, pool 6, dilation 2, se 0.25
    kernel 5, filters  64, pool 4, dilation 2, se 0.25
    kernel 3, filters  96, pool 2, dilation 1, se 0.25
    kernel 3, filters 128, pool 2, dilation 1, se 0.10
    kernel 3, filters 256, pool 1, dilation 1, se 0.10

with ASPP depth 512 and rates 6/12/18, the classifier squeeze ratio 0.25, and the
head shape their notebook specifies: a pointwise convolution with tanh, average
pooling to the segment length, then a pointwise segment classifier.

Three departures, all forced by this corpus rather than chosen, and all of which
make this a fair port and not a reproduction of their result:

  * Signals. Their Model 3 reads SpO2, nasal pressure, oronasal thermocouple, EEG
    and RIPsum. iSLEEPS has no separate thermocouple, so four of the five are used:
    SpO2, airflow, summed respiratory effort, and one EEG derivation (C4:M1).
  * Sampling rate. Theirs is 32 Hz for every signal, which is kept: the
    cardiorespiratory channels are carried up from 25 Hz and the EEG down from
    100 Hz by polyphase resampling.
  * Respiratory labels. Theirs is a three-class decision every second, none against
    hypopnea against apnea. iSLEEPS is scored per 30-second epoch, so the
    respiratory head predicts the same per-epoch binary label MM-Net does. This is
    the adaptation that makes the two models comparable at all, and it is also the
    one that takes away the finer output their design was built for.

Their 877-recording suspected-OSA cohort is not this cohort, so nothing here
speaks to the numbers in their paper. It answers only the question the review
asked: how their architecture performs on these patients, on these folds.

  KMP_DUPLICATE_LIB_OK=TRUE python run_huttunen.py
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample_poly
from sklearn.metrics import (accuracy_score, average_precision_score,
                             cohen_kappa_score, f1_score, roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import mmnet_core as C          # noqa: E402

MM = os.path.join(REPO, "data", "multimodal")
OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final")
FS = 32                         # their sampling rate, kept
SAMP = FS * 30                  # 960 samples per 30 s epoch
SEQ = 20                        # epochs per segment; SEQ*SAMP must divide 768
EPOCHS, BS, LR = 25, 8, 1e-3
SEEDS = [42, 1, 7]
DEV = C.DEV

# (kernel, filters, pool, dilation, se_ratio) -- DEFAULT_BLOCK_ARGS, verbatim
BLOCKS = [(5, 32, 8, 2, 0.25), (5, 48, 6, 2, 0.25), (5, 64, 4, 2, 0.25),
          (3, 96, 2, 1, 0.25), (3, 128, 2, 1, 0.10), (3, 256, 1, 1, 0.10)]
ASPP_DEPTH, ASPP_RATES, CLF_SE = 512, (6, 12, 18), 0.25

CARD_SPO2, CARD_FLOW, CARD_EFFORT = 5, 1, 4     # ECG Flow Thorax Abdomen Effort SpO2 Pulse
EEG_C4 = 0


# ----------------------------------------------------------------------
# architecture
# ----------------------------------------------------------------------
class SE(nn.Module):
    """Squeeze-and-excitation over channels, ratio as in their blocks."""

    def __init__(self, ch, ratio):
        super().__init__()
        red = max(1, int(ch * ratio))
        self.fc1 = nn.Conv1d(ch, red, 1)
        self.fc2 = nn.Conv1d(red, ch, 1)

    def forward(self, x):
        s = x.mean(-1, keepdim=True)
        s = torch.relu(self.fc1(s))
        return x * torch.sigmoid(self.fc2(s))


class Block(nn.Module):
    """conv-BN-relu, conv-BN-relu, SE, maxpool. The skip is taken before SE."""

    def __init__(self, cin, k, f, pool, dil, se):
        super().__init__()
        pad = dil * (k - 1) // 2
        self.c0 = nn.Conv1d(cin, f, k, dilation=dil, padding=pad, bias=False)
        self.b0 = nn.BatchNorm1d(f)
        self.c1 = nn.Conv1d(f, f, k, dilation=dil, padding=pad, bias=False)
        self.b1 = nn.BatchNorm1d(f)
        self.se = SE(f, se) if se else None
        self.pool = nn.MaxPool1d(pool) if pool > 1 else None

    def forward(self, x):
        x = torch.relu(self.b0(self.c0(x)))
        skip = torch.relu(self.b1(self.c1(x)))
        out = self.se(skip) if self.se is not None else skip
        if self.pool is not None:
            out = self.pool(out)
        return out, skip


class ASPP(nn.Module):
    """Atrous spatial pyramid pooling, depth and rates as released."""

    def __init__(self, cin, depth=ASPP_DEPTH, rates=ASPP_RATES):
        super().__init__()
        self.one = nn.Sequential(nn.Conv1d(cin, depth, 1, bias=False),
                                 nn.BatchNorm1d(depth), nn.ReLU())
        self.dil = nn.ModuleList(
            nn.Sequential(nn.Conv1d(cin, depth, 3, dilation=r, padding=r, bias=False),
                          nn.BatchNorm1d(depth), nn.ReLU()) for r in rates)
        self.pool = nn.Sequential(nn.Conv1d(cin, depth, 1, bias=False),
                                  nn.BatchNorm1d(depth), nn.ReLU())
        self.proj = nn.Sequential(nn.Conv1d(depth * (2 + len(rates)), depth, 1, bias=False),
                                  nn.BatchNorm1d(depth), nn.ReLU())

    def forward(self, x):
        parts = [self.one(x)] + [d(x) for d in self.dil]
        g = self.pool(x.mean(-1, keepdim=True)).expand(-1, -1, x.shape[-1])
        return self.proj(torch.cat(parts + [g], 1))


class Head(nn.Module):
    """SE, pointwise tanh convolution, average-pool to the segment, pointwise classifier."""

    def __init__(self, cin, n_cls, samples_per_segment):
        super().__init__()
        self.se = SE(cin, CLF_SE)
        self.dense = nn.Conv1d(cin, n_cls, 1)
        self.pool = nn.AvgPool1d(samples_per_segment)
        self.segment = nn.Conv1d(n_cls, n_cls, 1)

    def forward(self, x):
        x = torch.tanh(self.dense(self.se(x)))
        return self.segment(self.pool(x)).transpose(1, 2)      # [B, segments, n_cls]


class HuttunenUTime(nn.Module):
    def __init__(self, cin, n_stage=5):
        super().__init__()
        self.enc = nn.ModuleList()
        prev = cin
        for k, f, p, d, se in BLOCKS:
            self.enc.append(Block(prev, k, f, p, d, se))
            prev = f
        self.aspp = ASPP(prev)
        prev = ASPP_DEPTH
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        skips = [f for _, f, _, _, _ in BLOCKS][:-1]           # residuals[:-1]
        for (k, f, p, d, se), sk in zip(reversed(BLOCKS[:-1]), reversed(skips)):
            self.up.append(nn.Sequential(nn.Upsample(scale_factor=p, mode="nearest"),
                                         nn.Conv1d(prev, f, 2, padding=1, bias=False),
                                         nn.BatchNorm1d(f), nn.ReLU()))
            self.dec.append(Block(f + sk, k, f, 1, 1, None))
            prev = f
        self.stage_head = Head(prev, n_stage, SAMP)
        self.apnea_head = Head(prev, 1, SAMP)

    def forward(self, x):
        skips = []
        for b in self.enc:
            x, s = b(x)
            skips.append(s)
        x = self.aspp(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips[:-1])):
            x = up(x)[..., :s.shape[-1]]
            x, _ = dec(torch.cat([x, s], 1))
        return self.stage_head(x), self.apnea_head(x).squeeze(-1)


# ----------------------------------------------------------------------
# data at 32 Hz
# ----------------------------------------------------------------------
def load():
    """{sid: (x[n, 4, 960], y[n], a[n])} -- SpO2, airflow, effort, C4 at 32 Hz."""
    data = {}
    for f in sorted(glob.glob(os.path.join(MM, "SN*.npz")),
                    key=lambda p: int(os.path.basename(p)[2:-4])):
        sid = int(os.path.basename(f)[2:-4])
        if sid in C.DUP or sid not in C.DATA:
            continue
        z = np.load(f)
        card = z["card"].astype(np.float32)          # [n, 7, 750] at 25 Hz
        eeg = z["eeg"].astype(np.float32)            # [n, 7, 3000] at 100 Hz
        n = len(card)
        c = card[:, [CARD_SPO2, CARD_FLOW, CARD_EFFORT], :]
        c32 = resample_poly(c, FS, 25, axis=-1).astype(np.float32)
        e32 = resample_poly(eeg[:, [EEG_C4], :], FS, 100, axis=-1).astype(np.float32)
        x = np.concatenate([c32, e32], axis=1)       # [n, 4, 960]
        assert x.shape[-1] == SAMP, x.shape
        mu = x.mean(axis=(0, 2), keepdims=True)
        sd = x.std(axis=(0, 2), keepdims=True) + 1e-6
        data[sid] = ((x - mu) / sd, z["y"].astype(np.int64), z["apnea"].astype(np.float32))
        if n != len(C.DATA[sid][2]):
            raise ValueError("SN%d epoch mismatch" % sid)
    return data


def seqs(data, subs):
    X, Y, A, M = [], [], [], []
    for s in subs:
        x, y, a = data[s]
        for i in range(0, len(y), SEQ):
            xs, ys, as_ = x[i:i + SEQ], y[i:i + SEQ], a[i:i + SEQ]
            k = len(ys)
            if k < SEQ:
                pad = SEQ - k
                xs = np.concatenate([xs, np.zeros((pad,) + x.shape[1:], np.float32)])
                ys = np.concatenate([ys, np.zeros(pad, np.int64)])
                as_ = np.concatenate([as_, np.zeros(pad, np.float32)])
            X.append(xs.transpose(1, 0, 2).reshape(x.shape[1], -1))
            Y.append(ys); A.append(as_)
            M.append(np.concatenate([np.ones(k, np.float32), np.zeros(SEQ - k, np.float32)]))
    return (torch.tensor(np.stack(X)), torch.tensor(np.stack(Y)),
            torch.tensor(np.stack(A)), torch.tensor(np.stack(M)))


@torch.no_grad()
def infer(model, X, bs=8):
    model.eval()
    so, ao = [], []
    for i in range(0, len(X), bs):
        s, a = model(X[i:i + bs].to(DEV))
        so.append(s.softmax(-1).cpu().numpy()); ao.append(torch.sigmoid(a).cpu().numpy())
    return np.concatenate(so), np.concatenate(ao)


def run_fold(data, tr, va, te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Ytr, Atr, Mtr = seqs(data, tr)
    Xva, Yva, _, Mva = seqs(data, va)
    model = HuttunenUTime(cin=Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    cw = C.sqrt_cw(tr)
    ce = nn.CrossEntropyLoss(weight=cw, reduction="none")
    pos = float((1 - np.concatenate([data[s][2] for s in tr]).mean())
                / max(1e-6, np.concatenate([data[s][2] for s in tr]).mean()))
    bce = nn.BCEWithLogitsLoss(reduction="none",
                               pos_weight=torch.tensor([pos], device=DEV))
    best, best_state, bad = -1, None, 0
    N = len(Xtr)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(N)
        for i in range(0, N - BS + 1, BS):
            idx = perm[i:i + BS]
            x = Xtr[idx].to(DEV); y = Ytr[idx].to(DEV)
            a = Atr[idx].to(DEV); m = Mtr[idx].to(DEV).reshape(-1)
            s_o, a_o = model(x)
            ls = (ce(s_o.reshape(-1, 5), y.reshape(-1)) * m).sum() / m.sum().clamp(min=1)
            la = (bce(a_o.reshape(-1), a.reshape(-1)) * m).sum() / m.sum().clamp(min=1)
            loss = ls + la
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sch.step()
        sp, _ = infer(model, Xva)
        mk = Mva.numpy().reshape(-1) > 0
        acc = (sp.reshape(-1, 5).argmax(1)[mk] == Yva.numpy().reshape(-1)[mk]).mean()
        if acc > best:
            best, bad = acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
    model.load_state_dict(best_state)

    Xte, Yte, Ate, Mte = seqs(data, te)
    sp, ap = infer(model, Xte)
    mk = Mte.numpy().reshape(-1) > 0
    yt = Yte.numpy().reshape(-1)[mk]
    yp = sp.reshape(-1, 5).argmax(1)[mk]
    at = Ate.numpy().reshape(-1)[mk]
    asc = ap.reshape(-1)[mk]
    return dict(acc=float(accuracy_score(yt, yp)),
                mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                kappa=float(cohen_kappa_score(yt, yp)),
                auc=float(roc_auc_score(at, asc)) if len(np.unique(at)) > 1 else float("nan"),
                ap=float(average_precision_score(at, asc)) if len(np.unique(at)) > 1 else float("nan"))


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "huttunen.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    data = load()
    print("loaded %d patients at %d Hz, %d channels" % (len(data), FS, next(iter(data.values()))[0].shape[1]))
    probe = HuttunenUTime(cin=4)
    print("parameters: %s" % format(sum(p.numel() for p in probe.parameters()), ","))
    del probe
    t0 = time.time()

    # Folds are checkpointed individually. The first run of this script lost three
    # folds of a seed to a host-memory failure because only whole seeds were saved,
    # and each fold is several minutes.
    partial = res.setdefault("_partial", {})
    for seed in SEEDS:
        key = "seed|%d" % seed
        if key in res:
            print("[skip]", key, flush=True); continue
        for fi, (tr_all, te) in enumerate(C.FOLDS):
            pkey = "%d|%d" % (seed, fi)
            if pkey in partial:
                continue
            rng = np.random.RandomState(100 + fi)
            tr_all = list(tr_all); rng.shuffle(tr_all)
            nv = max(10, len(tr_all) // 9)
            va, tr = tr_all[:nv], tr_all[nv:]
            r = run_fold(data, tr, va, te, seed)
            partial[pkey] = r
            json.dump(res, open(path, "w"), indent=1)
            print("  seed %d fold %d  acc %.4f  kappa %.4f  auc %.4f  [%.1f min]"
                  % (seed, fi, r["acc"], r["kappa"], r["auc"], (time.time() - t0) / 60),
                  flush=True)
        folds = [partial["%d|%d" % (seed, fi)] for fi in range(len(C.FOLDS))]
        res[key] = {k: [f[k] for f in folds] for k in ("acc", "mf1", "kappa", "auc", "ap")}
        json.dump(res, open(path, "w"), indent=1)

    # ---- against MM-Net on the same fold-means -----------------------------
    fm = json.load(open(os.path.join(OUT, "final_model.json")))
    have = [s for s in SEEDS if "seed|%d" % s in res]
    if not have:
        print("\nno seed completed; nothing to compare")
        return
    if len(have) < len(SEEDS):
        print("\ncomparing on %d of %d seeds (%s); the rest did not finish"
              % (len(have), len(SEEDS), have))
    from scipy.stats import wilcoxon
    print("\n%-10s %10s %10s %10s %8s" % ("metric", "Huttunen", "MM-Net", "delta", "p"))
    summary = {"seeds": have}
    for m in ("acc", "mf1", "kappa", "auc", "ap"):
        h = np.mean(np.asarray([res["seed|%d" % s][m] for s in have], float), axis=0)
        j = np.mean(np.asarray([fm["final|%d" % s][m] for s in have], float), axis=0)
        p = wilcoxon(j, h).pvalue
        summary[m] = dict(huttunen=round(float(h.mean()), 4), mmnet=round(float(j.mean()), 4),
                          delta=round(float((j - h).mean()), 4), p=round(float(p), 4),
                          folds_mmnet_wins=int(((j - h) > 0).sum()))
        print("%-10s %10.4f %10.4f %+10.4f %8.4f" % (m, h.mean(), j.mean(), (j - h).mean(), p))

    res["_summary"] = summary
    res["_note"] = ("architecture ported from github.com/rikuhuttunen/psg-simultscoring-models; "
                    "four of their Model 3 signals (no thermocouple in iSLEEPS), 32 Hz as "
                    "specified, respiratory head adapted to this corpus's per-epoch binary label")
    json.dump(res, open(path, "w"), indent=1)
    print("\nsaved -> %s  [%.1f min]" % (path, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
