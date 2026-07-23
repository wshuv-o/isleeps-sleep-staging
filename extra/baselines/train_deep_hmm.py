"""
train_deep_hmm.py — best-model attempt: deep CNN+BiLSTM (GPU) + HMM Viterbi decoding.

Trains StagingSeqNet on the full cohort (overlap sequences + EEG augmentation, GPU),
then for each test subject extracts per-epoch class probabilities in temporal order and
decodes them through an HMM transition model estimated from the training hypnograms
(the same +2 pt trick that lifted the classical ensemble). Reports deep-argmax vs deep+HMM.

  KMP_DUPLICATE_LIB_OK=TRUE d:/EEG-TransNet/testenv/python.exe train_deep_hmm.py --epochs 40
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
from datasets import load_subject, make_folds, list_subjects, CLASS_NAMES, CHANNELS  # noqa
from datasets_seq import SequenceDataset  # noqa
from staging_seq import StagingSeqNet  # noqa

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
torch.backends.cudnn.benchmark = True
L = 20


def set_seed(s): np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def augment_batch(x, noise=0.05, scale=0.15, ch_drop=0.1, tmask=0.10):
    B, Ln, C, T = x.shape
    x = x * torch.empty(B, 1, C, 1, device=x.device).uniform_(1 - scale, 1 + scale)
    x = x + torch.randn_like(x) * noise
    if C > 1 and ch_drop > 0:
        keep = (torch.rand(B, 1, C, 1, device=x.device) > ch_drop).float()
        keep = torch.where(keep.sum(2, keepdim=True) == 0, torch.ones_like(keep), keep)
        x = x * keep
    if tmask > 0:
        w = int(T * tmask)
        if w > 0:
            st = torch.randint(0, T - w + 1, (1,), device=x.device).item(); x[..., st:st + w] = 0.0
    return x


def transition_matrix(seqs, n=5, eps=1.0):
    A = np.full((n, n), eps); pi = np.full(n, eps)
    for y in seqs:
        pi[y[0]] += 1
        for a, b in zip(y[:-1], y[1:]): A[a, b] += 1
    A /= A.sum(1, keepdims=True); pi /= pi.sum()
    return A, pi


def viterbi(log_e, log_A, log_pi):
    T, S = log_e.shape
    dp = np.zeros((T, S)); bp = np.zeros((T, S), int); dp[0] = log_pi + log_e[0]
    for t in range(1, T):
        sc = dp[t - 1][:, None] + log_A; bp[t] = sc.argmax(0); dp[t] = sc.max(0) + log_e[t]
    path = np.zeros(T, int); path[-1] = dp[-1].argmax()
    for t in range(T - 2, -1, -1): path[t] = bp[t + 1, path[t + 1]]
    return path


@torch.no_grad()
def subject_probs(model, sid, channels, device):
    """Per-epoch softmax probabilities for one subject, in temporal order. -> probs[n,5], y[n]."""
    x, y = load_subject(sid, channels=channels, normalize=True)      # [n,C,3000]
    n = len(y); pad = (-n) % L
    xp = np.concatenate([x, np.zeros((pad,) + x.shape[1:], np.float32)], 0) if pad else x
    seqs = xp.reshape(-1, L, x.shape[1], x.shape[2])                 # [n_seq,L,C,T]
    probs = []
    for i in range(0, len(seqs), 16):
        xb = torch.tensor(seqs[i:i + 16], device=device)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logit = model(xb)                                        # [b,L,5]
        probs.append(torch.softmax(logit.float(), -1).reshape(-1, 5).cpu().numpy())
    probs = np.concatenate(probs)[:n]
    return probs, y


def metrics(y, p):
    return {"acc": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "per_class_f1": f1_score(y, p, average=None, labels=list(range(5)), zero_division=0).tolist(),
            "kappa": float(cohen_kappa_score(y, p)),
            "confusion": confusion_matrix(y, p, labels=list(range(5))).tolist()}


def train_fold(fold, train_core, val_s, test_s, channels, args, device):
    tr = SequenceDataset(train_core, seq_len=L, stride=L // 2, channels=channels)   # overlap
    counts = tr.class_counts()
    w = torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, pin_memory=True)
    model = StagingSeqNet(in_ch=len(channels), dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(weight=w, reduction="none")
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # HMM transition model from training hypnograms
    A, pi = transition_matrix([load_subject(s)[1] for s in train_core])
    logA, logpi = np.log(A + 1e-12), np.log(pi + 1e-12)

    best_val = -1; best = None
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y, m in tqdm(tl, desc=f"fold{fold} ep{ep:02d}/{args.epochs}", leave=False, ncols=90, mininterval=0.5):
            x, y, m = x.to(device), y.to(device), m.to(device)
            x = augment_batch(x)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(x)
                loss = crit(logits.reshape(-1, 5), y.reshape(-1))
                loss = (loss * m.reshape(-1)).sum() / m.sum().clamp(min=1)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update()
        sched.step()
        # validation (deep-argmax) for model selection
        vy, vp = [], []
        for s in val_s:
            pr, yy = subject_probs(model, s, channels, device); vy.append(yy); vp.append(pr.argmax(1))
        vmf1 = f1_score(np.concatenate(vy), np.concatenate(vp), average="macro", zero_division=0)
        if vmf1 > best_val:
            best_val = vmf1; best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"  fold{fold} ep{ep:02d} val_mF1={vmf1:.3f}{' *' if best_val==vmf1 else ''}")

    model.load_state_dict(best)
    # test: deep vs deep+HMM
    ty, tp_deep, tp_hmm = [], [], []
    for s in test_s:
        pr, yy = subject_probs(model, s, channels, device)
        ty.append(yy); tp_deep.append(pr.argmax(1)); tp_hmm.append(viterbi(np.log(pr + 1e-12), logA, logpi))
    ty = np.concatenate(ty)
    md = metrics(ty, np.concatenate(tp_deep)); mh = metrics(ty, np.concatenate(tp_hmm))
    print(f"  fold{fold}: deep acc={md['acc']:.3f} mF1={md['macro_f1']:.3f} | "
          f"+HMM acc={mh['acc']:.3f} mF1={mh['macro_f1']:.3f} k={mh['kappa']:.3f}")
    return md, mh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", default="C4:M1,C3:M2,O2:M1,O1:M2")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--val-subj", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); set_seed(args.seed)
    channels = [c.strip() for c in args.channels.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    subs = list_subjects(); folds = make_folds(subs, 5, seed=args.seed)
    rng = np.random.RandomState(args.seed)
    print(f"device={device} | subjects={len(subs)} | deep CNN+BiLSTM + HMM")
    deep_res, hmm_res = [], []
    for k, (tr_s, te_s) in enumerate(folds):
        vs = sorted(rng.choice(tr_s, size=args.val_subj, replace=False).tolist())
        tc = [s for s in tr_s if s not in vs]
        print(f"\n=== Fold {k} | train {len(tc)} / val {len(vs)} / test {len(te_s)} ===")
        md, mh = train_fold(k, tc, vs, te_s, channels, args, device)
        deep_res.append(md); hmm_res.append(mh)

    def summ(name, res):
        acc = np.array([r["acc"] for r in res]); mf1 = np.array([r["macro_f1"] for r in res])
        kap = np.array([r["kappa"] for r in res]); pcf = np.array([r["per_class_f1"] for r in res]).mean(0)
        print(f"{name:16s} acc={acc.mean():.4f}+-{acc.std():.4f} mF1={mf1.mean():.4f} kappa={kap.mean():.4f}  "
              + " ".join(f"{c}={v:.3f}" for c, v in zip(CLASS_NAMES, pcf)))
        return {"acc": float(acc.mean()), "acc_std": float(acc.std()), "macro_f1": float(mf1.mean()),
                "kappa": float(kap.mean()), "per_class_f1": pcf.tolist()}
    print("\n===== full-cohort deep results =====")
    out = {"deep": summ("deep (argmax)", deep_res), "deep_hmm": summ("deep + HMM", hmm_res)}
    json.dump(out, open(os.path.join(RESULTS, "deep_hmm_full.json"), "w"), indent=2)
    print("saved -> results/deep_hmm_full.json")


if __name__ == "__main__":
    main()
