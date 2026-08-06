"""
train.py — subject-independent CV training for iSLEEPS sleep staging.

Usage (always with the GPU env):
  KMP_DUPLICATE_LIB_OK=TRUE  d:/EEG-TransNet/testenv/python.exe train.py --fold 0 --epochs 30
  ... --all-folds            # full 5-fold subject-independent CV (E1)

Metrics: accuracy, macro-F1, per-class F1, Cohen's kappa, confusion matrix.
Class imbalance handled via inverse-frequency weighted cross-entropy.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
from datasets import EpochDataset, make_folds, list_subjects, CLASS_NAMES, CHANNELS  # noqa
from staging_cnn import StagingCNN, count_params  # noqa

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
torch.backends.cudnn.benchmark = True  # autotune convs for fixed [B,C,3000] shape


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            ps.append(logits.argmax(1).cpu().numpy()); ys.append(y.numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    return {
        "acc": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "per_class_f1": f1_score(y, p, average=None, labels=list(range(5)), zero_division=0).tolist(),
        "kappa": float(cohen_kappa_score(y, p)),
        "confusion": confusion_matrix(y, p, labels=list(range(5))).tolist(),
        "n": int(len(y)),
    }


def train_fold(fold, train_sids, test_sids, channels, args, device):
    tr = EpochDataset(train_sids, channels=channels)
    te = EpochDataset(test_sids, channels=channels)
    counts = tr.class_counts()
    weights = torch.tensor((counts.sum() / (len(counts) * np.maximum(counts, 1))),
                           dtype=torch.float32, device=device)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=False,
                    num_workers=0, pin_memory=True)
    vl = DataLoader(te, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

    model = StagingCNN(in_ch=len(channels), dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(weight=weights)
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = {"macro_f1": -1}
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        pbar = tqdm(tl, desc=f"fold{fold} ep{ep:02d}/{args.epochs}", leave=False,
                    ncols=90, mininterval=0.5)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(y)
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        sched.step()
        m = evaluate(model, vl, device)
        tag = ""
        if m["macro_f1"] > best["macro_f1"]:
            best = {**m, "epoch": ep}; tag = " *"
        print(f"  fold{fold} ep{ep:02d} loss={tot/len(tr):.3f} "
              f"acc={m['acc']:.3f} mF1={m['macro_f1']:.3f} kappa={m['kappa']:.3f}{tag}")
    print(f"  fold{fold} BEST: acc={best['acc']:.3f} mF1={best['macro_f1']:.3f} "
          f"kappa={best['kappa']:.3f} (ep{best['epoch']})  "
          f"per-class F1={[round(f,3) for f in best['per_class_f1']]}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--channels", default="C4:M1", help="comma list; default single C4:M1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="cnn")
    args = ap.parse_args()
    set_seed(args.seed)

    channels = [c.strip() for c in args.channels.split(",")] if args.channels else CHANNELS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RESULTS, exist_ok=True)
    subs = list_subjects()
    folds = make_folds(subs, n_splits=5, seed=args.seed)
    print(f"device={device} | channels={channels} | params="
          f"{count_params(StagingCNN(in_ch=len(channels))):,} | subjects={len(subs)}")

    todo = list(range(5)) if args.all_folds else [args.fold if args.fold is not None else 0]
    results = []
    for k in todo:
        tr_s, te_s = folds[k]
        print(f"\n=== Fold {k} | train {len(tr_s)} subj, test {len(te_s)} subj {te_s} ===")
        results.append(train_fold(k, tr_s, te_s, channels, args, device))

    if results:
        accs = [r["acc"] for r in results]; mf1 = [r["macro_f1"] for r in results]
        kap = [r["kappa"] for r in results]
        pcf = np.array([r["per_class_f1"] for r in results])
        print(f"\n===== {len(results)} fold(s) summary =====")
        print(f"  accuracy : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"  macro-F1 : {np.mean(mf1):.4f} +/- {np.std(mf1):.4f}")
        print(f"  kappa    : {np.mean(kap):.4f} +/- {np.std(kap):.4f}")
        print("  per-class F1: " + "  ".join(
            f"{n}={pcf[:,i].mean():.3f}" for i, n in enumerate(CLASS_NAMES)))
        out = os.path.join(RESULTS, f"{args.tag}_{'all' if args.all_folds else todo[0]}.json")
        json.dump({"args": vars(args), "channels": channels, "folds": results,
                   "mean": {"acc": float(np.mean(accs)), "macro_f1": float(np.mean(mf1)),
                            "kappa": float(np.mean(kap))}}, open(out, "w"), indent=2)
        print(f"  saved -> {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
