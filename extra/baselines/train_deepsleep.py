"""
train_deepsleep.py — two-stage DeepSleepNet-style training, subject-independent CV,
with HONEST validation-based model selection (no test peeking).

  Stage 1: pretrain the dual-res encoder on per-epoch classification with
           class-balanced sampling (WeightedRandomSampler).
  Stage 2: train the full encoder+residual-BiLSTM on epoch sequences (masked
           weighted loss). Each epoch we score a held-out VALIDATION set (subjects
           split out of train); we report the TEST metrics at the best-val epoch.

  KMP_DUPLICATE_LIB_OK=TRUE d:/EEG-TransNet/testenv/python.exe train_deepsleep.py --all-folds
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
from datasets import EpochDataset, make_folds, list_subjects, CLASS_NAMES, CHANNELS  # noqa
from datasets_seq import SequenceDataset  # noqa
from deepsleep import DeepSleepSeq  # noqa

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
torch.backends.cudnn.benchmark = True


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def class_weights(counts, device):
    return torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)),
                        dtype=torch.float32, device=device)


@torch.no_grad()
def eval_seq(model, loader, device):
    model.eval(); ys, ps = [], []
    for x, y, m in loader:
        p = model(x.to(device)).argmax(-1).cpu().numpy()
        m = m.numpy().astype(bool); y = y.numpy()
        ys.append(y[m]); ps.append(p[m])
    y = np.concatenate(ys); p = np.concatenate(ps)
    return {"acc": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "per_class_f1": f1_score(y, p, average=None, labels=list(range(5)), zero_division=0).tolist(),
            "kappa": float(cohen_kappa_score(y, p)),
            "confusion": confusion_matrix(y, p, labels=list(range(5))).tolist(),
            "n": int(len(y))}


def pretrain_encoder(model, train_core, channels, args, device):
    ds = EpochDataset(train_core, channels=channels)
    counts = ds.class_counts()
    w = (counts.sum() / (len(counts) * np.maximum(counts, 1)))
    sample_w = w[ds.y]
    sampler = WeightedRandomSampler(torch.tensor(sample_w, dtype=torch.double), len(ds.y), replacement=True)
    dl = DataLoader(ds, batch_size=args.batch_pre, sampler=sampler, pin_memory=True)
    opt = torch.optim.Adam(list(model.encoder.parameters()) + list(model.proj.parameters())
                           + list(model.epoch_head.parameters()), lr=args.lr_pre, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    for ep in range(1, args.pre_epochs + 1):
        model.train(); tot = 0.0
        pbar = tqdm(dl, desc=f"  pretrain ep{ep:02d}/{args.pre_epochs}", leave=False, ncols=90, mininterval=0.5)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                loss = crit(model.classify_epoch(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item(); pbar.set_postfix(loss=f"{loss.item():.3f}")
        print(f"  [stage1] ep{ep:02d} balanced-CE loss={tot/len(dl):.3f}")


def train_fold(fold, train_core, val_s, test_s, channels, args, device):
    model = DeepSleepSeq(in_ch=len(channels), dropout=args.dropout).to(device)
    print(f"  stage 1: pretrain encoder on {len(train_core)} subj (balanced)")
    pretrain_encoder(model, train_core, channels, args, device)

    tr = SequenceDataset(train_core, seq_len=args.seq_len, stride=args.seq_len // 2, channels=channels)
    va = SequenceDataset(val_s, seq_len=args.seq_len, stride=args.seq_len, channels=channels)
    te = SequenceDataset(test_s, seq_len=args.seq_len, stride=args.seq_len, channels=channels)
    w = class_weights(tr.class_counts(), device)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, pin_memory=True)
    vl = DataLoader(va, batch_size=64, shuffle=False, pin_memory=True)
    el = DataLoader(te, batch_size=64, shuffle=False, pin_memory=True)

    # encoder gets a smaller LR (already pretrained), new seq layers full LR
    enc_p = list(model.encoder.parameters()) + list(model.proj.parameters())
    new_p = list(model.lstm.parameters()) + list(model.res.parameters()) + list(model.head.parameters())
    opt = torch.optim.Adam([{"params": enc_p, "lr": args.lr * 0.3},
                            {"params": new_p, "lr": args.lr}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(weight=w, reduction="none")
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    print(f"  stage 2: train seq | train {len(train_core)} / val {len(val_s)} / test {len(test_s)} subj")
    best_val = -1; best = None
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0; nb = 0
        pbar = tqdm(tl, desc=f"fold{fold} ep{ep:02d}/{args.epochs}", leave=False, ncols=90, mininterval=0.5)
        for x, y, m in pbar:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(x)
                loss = crit(logits.reshape(-1, 5), y.reshape(-1))
                loss = (loss * m.reshape(-1)).sum() / m.sum().clamp(min=1)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1; pbar.set_postfix(loss=f"{loss.item():.3f}")
        sched.step()
        vm = eval_seq(model, vl, device)
        tm = eval_seq(model, el, device)
        tag = ""
        if vm["macro_f1"] > best_val:
            best_val = vm["macro_f1"]; best = {**tm, "epoch": ep, "val_macro_f1": vm["macro_f1"]}; tag = " *"
        print(f"  fold{fold} ep{ep:02d} loss={tot/nb:.3f} | val mF1={vm['macro_f1']:.3f} "
              f"| test acc={tm['acc']:.3f} mF1={tm['macro_f1']:.3f} k={tm['kappa']:.3f}{tag}")
    print(f"  fold{fold} SELECTED (best val ep{best['epoch']}): test acc={best['acc']:.3f} "
          f"mF1={best['macro_f1']:.3f} kappa={best['kappa']:.3f} "
          f"per-class F1={[round(f,3) for f in best['per_class_f1']]}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--channels", default="C4:M1,C3:M2,O2:M1,O1:M2")
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--pre-epochs", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--batch-pre", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-pre", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--val-subj", type=int, default=5, help="train subjects held out for selection")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="deepsleep")
    args = ap.parse_args()
    set_seed(args.seed)

    channels = [c.strip() for c in args.channels.split(",")] if args.channels else CHANNELS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RESULTS, exist_ok=True)
    subs = list_subjects()
    folds = make_folds(subs, n_splits=5, seed=args.seed)
    rng = np.random.RandomState(args.seed)
    nparams = sum(p.numel() for p in DeepSleepSeq(in_ch=len(channels)).parameters())
    print(f"device={device} | channels={channels} | seq_len={args.seq_len} | "
          f"params={nparams:,} | subjects={len(subs)}")

    todo = list(range(5)) if args.all_folds else [args.fold if args.fold is not None else 0]
    results = []
    for k in todo:
        tr_s, te_s = folds[k]
        vs = sorted(rng.choice(tr_s, size=args.val_subj, replace=False).tolist())
        tc = [s for s in tr_s if s not in vs]
        print(f"\n=== Fold {k} | train {len(tc)} / val {len(vs)} {vs} / test {len(te_s)} {te_s} ===")
        results.append(train_fold(k, tc, vs, te_s, channels, args, device))

    accs = [r["acc"] for r in results]; mf1 = [r["macro_f1"] for r in results]
    kap = [r["kappa"] for r in results]; pcf = np.array([r["per_class_f1"] for r in results])
    print(f"\n===== {len(results)} fold(s) summary (val-selected, honest) =====")
    print(f"  accuracy : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  macro-F1 : {np.mean(mf1):.4f} +/- {np.std(mf1):.4f}")
    print(f"  kappa    : {np.mean(kap):.4f} +/- {np.std(kap):.4f}")
    print("  per-class F1: " + "  ".join(f"{n}={pcf[:,i].mean():.3f}" for i, n in enumerate(CLASS_NAMES)))
    out = os.path.join(RESULTS, f"{args.tag}_{'all' if args.all_folds else todo[0]}.json")
    json.dump({"args": vars(args), "channels": channels, "folds": results,
               "mean": {"acc": float(np.mean(accs)), "macro_f1": float(np.mean(mf1)),
                        "kappa": float(np.mean(kap))}}, open(out, "w"), indent=2)
    print(f"  saved -> {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
