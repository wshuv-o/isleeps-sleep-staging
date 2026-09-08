"""Arm C -- CBraMod fine-tuned end to end inside every fold.

Mirrors mmnet_core.train_fold exactly (AdamW, cosine schedule, sqrt class
weights, pos-weighted BCE, gradient clipping at 5.0, early stopping on mean
validation subject accuracy with patience 8) and changes only two things, both
forced by fine-tuning a 4.9 M-parameter encoder on raw signal:

  * the data path streams (see stream.py -- the pre-materialised window tensor
    would be ~14 GB for raw signal), and
  * the encoder gets a lower learning rate than the rest of the model, which is
    standard for fine-tuning a pretrained backbone and without which the
    pretrained weights are destroyed in the first few steps.

--pretrained/--random selects between Arm C and its randomly-initialised
control, from the same code path.

Checkpoints after every fold, so a killed session loses at most one fold.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import mmnet_core as C          # noqa: E402
import stream                   # noqa: E402
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,        # noqa: E402
                             roc_auc_score, average_precision_score)

CKPT = os.path.join(HERE, "cbramod.pth")
OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "foundation")


def build_model(pretrained, seed, lr_head, lr_enc, wd, drop=0.3):
    torch.manual_seed(seed)
    # n_eeg=1: eeg_enc is replaced wholesale, so the default 188-wide FeatMLP
    # would only allocate an unused matrix.
    model = C.MMFeatureNet(n_eeg=1, fusion="concat", bypass=True, temporal="lstm").to(C.DEV)
    enc = stream.CBraModEncoder(C, CKPT, pretrained=pretrained, d=128, drop=drop,
                                seed=seed, trainable=True).to(C.DEV)
    model.eeg_enc = enc
    backbone = list(enc.encoder.parameters())
    bb_ids = {id(p) for p in backbone}
    rest = [p for p in model.parameters() if id(p) not in bb_ids]
    opt = torch.optim.AdamW([{"params": backbone, "lr": lr_enc},
                             {"params": rest, "lr": lr_head}], weight_decay=wd)
    return model, opt


@torch.no_grad()
def infer_subject(model, store, sid, bs):
    """Per-subject inference, streamed. Mirrors mmnet_core.subj_infer."""
    model.eval()
    L = C.L
    n = len(store.y[sid])
    pad = (-n) % L
    raw = store.x[sid]
    fc = store.fcard[sid]
    fe = store.epochs_to_model(raw)
    if pad:
        fe = np.concatenate([fe, np.zeros((pad, fe.shape[1]), np.float32)])
        fc = np.concatenate([fc, np.zeros((pad, fc.shape[1]), np.float32)])
    fe = fe.reshape(-1, L, fe.shape[1])
    fc = fc.reshape(-1, L, fc.shape[1])
    sp, ap = [], []
    for i in range(0, len(fe), bs):
        e = torch.tensor(fe[i:i + bs], device=C.DEV)
        c = torch.tensor(fc[i:i + bs], device=C.DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_o, a_o = model(e, c)
        sp.append(torch.softmax(s_o.float(), -1).reshape(-1, 5).cpu().numpy())
        ap.append(torch.sigmoid(a_o.float()).reshape(-1).cpu().numpy())
    return np.concatenate(sp)[:n], np.concatenate(ap)[:n]


def train_fold(store, tr, va, seed, args):
    model, opt = build_model(args.pretrained, seed, args.lr_head, args.lr_enc, args.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss(weight=C.sqrt_cw(tr), reduction="none")
    ac = np.zeros(2, np.int64)
    for s in tr:
        ac += np.bincount(C.DATA[s][3], minlength=2)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(
        [ac[0] / max(1, ac[1])], dtype=torch.float32, device=C.DEV))

    idx = store.windows(tr, C.L // 2)
    rng = np.random.RandomState(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(idx))
        t0 = time.time()
        # Effective batch is bs * accum, kept at 32 to match mmnet_core.train_fold.
        # It is split because a 32-window batch is 640 sleep epochs of raw signal
        # through a 12-layer transformer with gradients: that peaks at 16.2 GB on a
        # 17 GB card and collapses to 0.61 s/window, against 0.0123 s/window at
        # bs=16 (8.2 GB). A 50x cliff, entirely from memory pressure.
        eff = args.bs * args.accum
        opt.zero_grad()
        for i in range(0, len(order) - eff + 1, eff):
            for a_i in range(args.accum):
                lo = i + a_i * args.bs
                batch = [idx[j] for j in order[lo:lo + args.bs]]
                Fe, Fc, Y, Ap, Mk = store.batch(batch, C.DEV)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    s_o, a_o = model(Fe, Fc)
                    m = Mk.reshape(-1)
                    ls = (ce(s_o.reshape(-1, 5).float(), Y.reshape(-1)) * m).sum() / m.sum().clamp(min=1)
                    la = (bce(a_o.reshape(-1).float(), Ap.reshape(-1)) * m).sum() / m.sum().clamp(min=1)
                    loss = (ls + la) / args.accum
                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); opt.zero_grad()
        sch.step()
        acc = float(np.mean([(infer_subject(model, store, s, args.bs)[0].argmax(1)
                              == C.DATA[s][2]).mean() for s in va]))
        print("      epoch %2d  val acc %.4f  [%.1f min]" % (ep, acc, (time.time() - t0) / 60),
              flush=True)
        if acc > best:
            best, bad = acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    ap.add_argument("--random", dest="pretrained", action="store_false")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="subset of folds")
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)   # 32 exhausts VRAM; see train_fold
    ap.add_argument("--accum", type=int, default=2)  # bs*accum = 32, matching mmnet_core
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-enc", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    args = ap.parse_args()

    name = "C_finetuned" if args.pretrained else "C_finetuned_random"
    os.makedirs(OUT, exist_ok=True)
    cache_path = os.path.join(OUT, "arm_c.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    print("building raw store ...", flush=True)
    store = stream.RawStore(C)
    gb = sum(v.nbytes for v in store.x.values()) / 1e9
    print("raw store: %d subjects, %.2f GB fp16 (CPU)" % (len(store.subs), gb), flush=True)

    for seed in args.seeds:
        # FOLDS is fixed at module level; the seed varies training only, exactly as
        # run_10fold does. The validation split and the HMM priors are reproduced
        # verbatim so Arm C is scored the same way as every other arm.
        for fi, (tr_all, te) in enumerate(C.FOLDS):
            if args.folds is not None and fi not in args.folds:
                continue
            key = "%s|%d|%d" % (name, seed, fi)
            if key in cache:
                print("[skip] %s" % key, flush=True)
                continue
            rng = np.random.RandomState(100 + fi)
            tr_all = list(tr_all); rng.shuffle(tr_all)
            nv = max(10, len(tr_all) // 9)
            va, tr = tr_all[:nv], tr_all[nv:]
            Am = np.ones((C.NC, C.NC)); pi = np.ones(C.NC)
            for s in tr:
                y = C.DATA[s][2]; pi[y[0]] += 1
                for x, z in zip(y[:-1], y[1:]):
                    Am[x, z] += 1
            A_log = np.log(Am / Am.sum(1, keepdims=True)); pi_log = np.log(pi / pi.sum())
            t0 = time.time()
            print("=== %s seed %d fold %d (%d train / %d val / %d test) ==="
                  % (name, seed, fi, len(tr), len(va), len(te)), flush=True)
            model = train_fold(store, tr, va, seed, args)
            yt, ph, ay, apr = [], [], [], []
            for s in te:
                sp, apn = infer_subject(model, store, s, args.bs)
                y = C.DATA[s][2]
                ph.append(C.hmm(A_log, pi_log, np.log(sp + C.EPS)))
                yt.append(y); ay.append(C.DATA[s][3]); apr.append(apn)
            yt, ph = np.concatenate(yt), np.concatenate(ph)
            ay, apr = np.concatenate(ay), np.concatenate(apr)
            cache[key] = {
                "arm": name, "seed": seed, "fold": fi,
                "acc": float(accuracy_score(yt, ph)),
                "mf1": float(f1_score(yt, ph, average="macro", zero_division=0)),
                "kappa": float(cohen_kappa_score(yt, ph)),
                "auc": float(roc_auc_score(ay, apr)) if len(np.unique(ay)) > 1 else float("nan"),
                "ap": float(average_precision_score(ay, apr)) if len(np.unique(ay)) > 1 else float("nan"),
                "minutes": (time.time() - t0) / 60,
            }
            json.dump(cache, open(cache_path, "w"), indent=1)
            print("    -> acc %.4f  AUC %.4f  [%.1f min]"
                  % (cache[key]["acc"], cache[key]["auc"], cache[key]["minutes"]), flush=True)
    print("done ->", cache_path)


if __name__ == "__main__":
    main()
