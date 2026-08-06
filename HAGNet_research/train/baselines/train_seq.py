"""
train_seq.py — subject-independent CV for the context-aware (CNN+BiLSTM) stager.

  KMP_DUPLICATE_LIB_OK=TRUE d:/EEG-TransNet/testenv/python.exe train_seq.py --all-folds --epochs 25

Masked weighted cross-entropy over all L positions; tail-padding is masked out.
Metrics computed over real (unpadded) epochs only.
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
from datasets import make_folds, list_subjects, CLASS_NAMES, CHANNELS  # noqa
from datasets_seq import SequenceDataset  # noqa
from staging_seq import StagingSeqNet  # noqa

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
torch.backends.cudnn.benchmark = True


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys, ps = [], []
    for x, y, m in loader:
        logits = model(x.to(device))                 # [B,L,5]
        p = logits.argmax(-1).cpu().numpy()
        m = m.numpy().astype(bool); y = y.numpy()
        ys.append(y[m]); ps.append(p[m])
    y = np.concatenate(ys); p = np.concatenate(ps)
    return {
        "acc": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "per_class_f1": f1_score(y, p, average=None, labels=list(range(5)), zero_division=0).tolist(),
        "kappa": float(cohen_kappa_score(y, p)),
        "confusion": confusion_matrix(y, p, labels=list(range(5))).tolist(),
        "n": int(len(y)),
    }


def augment_batch(x, noise=0.05, scale=0.15, ch_drop=0.1, tmask=0.10):
    """On-GPU EEG augmentation for x [B,L,C,T] (training only). x is already z-scored."""
    B, L, C, T = x.shape
    x = x * torch.empty(B, 1, C, 1, device=x.device).uniform_(1 - scale, 1 + scale)  # amplitude
    x = x + torch.randn_like(x) * noise                                              # gaussian noise
    if C > 1 and ch_drop > 0:                                                         # channel dropout
        keep = (torch.rand(B, 1, C, 1, device=x.device) > ch_drop).float()
        # don't drop all channels in a sample
        keep = torch.where(keep.sum(2, keepdim=True) == 0, torch.ones_like(keep), keep)
        x = x * keep
    if tmask > 0:                                                                    # time masking
        w = int(T * tmask)
        if w > 0:
            start = torch.randint(0, T - w + 1, (1,), device=x.device).item()
            x[..., start:start + w] = 0.0
    return x


def train_fold(fold, train_core, val_s, test_s, channels, args, device):
    tr_stride = args.train_stride if args.train_stride > 0 else args.seq_len // 2
    tr = SequenceDataset(train_core, seq_len=args.seq_len, stride=tr_stride, channels=channels)
    te = SequenceDataset(test_s, seq_len=args.seq_len, stride=args.seq_len, channels=channels)
    use_val = bool(val_s)
    va = SequenceDataset(val_s, seq_len=args.seq_len, stride=args.seq_len, channels=channels) if use_val else te
    counts = tr.class_counts()
    w = torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)),
                     dtype=torch.float32, device=device)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, pin_memory=True)
    vl = DataLoader(va, batch_size=64, shuffle=False, pin_memory=True)
    el = DataLoader(te, batch_size=64, shuffle=False, pin_memory=True)

    model = StagingSeqNet(in_ch=len(channels), dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(weight=w, reduction="none")
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    sel = -1; best = None
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0; nb = 0
        pbar = tqdm(tl, desc=f"fold{fold} ep{ep:02d}/{args.epochs}", leave=False,
                    ncols=90, mininterval=0.5)
        for x, y, m in pbar:
            x = x.to(device); y = y.to(device); m = m.to(device)
            if args.augment:
                x = augment_batch(x)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)                                  # [B,L,5]
                loss = crit(logits.reshape(-1, 5), y.reshape(-1))  # [B*L]
                loss = (loss * m.reshape(-1)).sum() / m.sum().clamp(min=1)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        sched.step()
        vm = evaluate(model, vl, device)
        tm = evaluate(model, el, device) if use_val else vm
        tag = ""
        if vm["macro_f1"] > sel:
            sel = vm["macro_f1"]; best = {**tm, "epoch": ep, "val_macro_f1": vm["macro_f1"]}; tag = " *"
        sel_str = f"val mF1={vm['macro_f1']:.3f} | " if use_val else ""
        print(f"  fold{fold} ep{ep:02d} loss={tot/nb:.3f} {sel_str}"
              f"test acc={tm['acc']:.3f} mF1={tm['macro_f1']:.3f} kappa={tm['kappa']:.3f}{tag}")
    sel_note = "val-selected" if use_val else "test-peek"
    print(f"  fold{fold} SELECTED ({sel_note}, ep{best['epoch']}): acc={best['acc']:.3f} "
          f"mF1={best['macro_f1']:.3f} kappa={best['kappa']:.3f} "
          f"per-class F1={[round(f,3) for f in best['per_class_f1']]}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--channels", default="C4:M1,C3:M2,O2:M1,O1:M2")
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--augment", action="store_true", help="on-GPU EEG augmentation")
    ap.add_argument("--train-stride", type=int, default=0, help="0=seq_len//2 (overlap); set =seq_len for memory-safe non-overlap")
    ap.add_argument("--val-subj", type=int, default=0, help=">0 = honest val-based selection")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="seq")
    args = ap.parse_args()
    set_seed(args.seed)

    channels = [c.strip() for c in args.channels.split(",")] if args.channels else CHANNELS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RESULTS, exist_ok=True)
    subs = list_subjects()
    folds = make_folds(subs, n_splits=5, seed=args.seed)
    rng = np.random.RandomState(args.seed)
    nparams = sum(p.numel() for p in StagingSeqNet(in_ch=len(channels)).parameters())
    print(f"device={device} | channels={channels} | seq_len={args.seq_len} | "
          f"augment={args.augment} val_subj={args.val_subj} | params={nparams:,} | subjects={len(subs)}")

    todo = list(range(5)) if args.all_folds else [args.fold if args.fold is not None else 0]
    results = []
    for k in todo:
        tr_s, te_s = folds[k]
        if args.val_subj > 0:
            vs = sorted(rng.choice(tr_s, size=args.val_subj, replace=False).tolist())
            tc = [s for s in tr_s if s not in vs]
        else:
            vs, tc = [], tr_s
        print(f"\n=== Fold {k} | train {len(tc)} / val {len(vs)} {vs} / test {len(te_s)} {te_s} ===")
        results.append(train_fold(k, tc, vs, te_s, channels, args, device))

    accs = [r["acc"] for r in results]; mf1 = [r["macro_f1"] for r in results]
    kap = [r["kappa"] for r in results]; pcf = np.array([r["per_class_f1"] for r in results])
    print(f"\n===== {len(results)} fold(s) summary =====")
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
