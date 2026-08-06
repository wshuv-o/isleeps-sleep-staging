"""
train_featseq.py -- data-efficient WINNER attempt: a bidirectional LSTM over the WHOLE
NIGHT of engineered features (188-dim/epoch, features_v2). Feeds the ensemble's good
features into a real long-range sequence model (unlike the ensemble's +-3 context + HMM),
while staying data-efficient (input is 188 numbers/epoch, not raw signal).
Subject-independent CV, class-balanced loss, saves out-of-fold probs for stacking.
  KMP_DUPLICATE_LIB_OK=TRUE d:/EEG-TransNet/testenv/python.exe train_featseq.py --folds 10
"""
import os, sys, json, argparse, glob
import numpy as np
import torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "models"))
from datasets import make_folds, DUPLICATE_DROP  # noqa
from features_v2 import extract_features_v2  # noqa

PROC7 = os.path.join(HERE, "data", "processed7")
RESULTS = os.path.join(HERE, "results")
FCACHE = os.path.join(HERE, "data", "featseq_cache")
FEAT = {}   # sid -> (F [n,188] float32 normalized, y)


def seed(s): np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def list7():
    s = [int(os.path.basename(p)[2:-4]) for p in glob.glob(os.path.join(PROC7, "SN*.npz"))]
    return sorted(x for x in s if x not in DUPLICATE_DROP)


def load_feats(sid):
    cf = os.path.join(FCACHE, f"SN{sid}.npz")
    if os.path.exists(cf):
        d = np.load(cf); FEAT[sid] = (d["F"], d["y"]); return
    d = np.load(os.path.join(PROC7, f"SN{sid}.npz"), allow_pickle=True)
    F, _ = extract_features_v2(d["x"].astype(np.float32), fs=100)
    F = np.nan_to_num(F).astype(np.float32); y = d["y"].astype(np.int64)
    os.makedirs(FCACHE, exist_ok=True); np.savez(cf, F=F, y=y)
    FEAT[sid] = (F, y)


def norm_all(train_subs):
    # standardize features using TRAIN statistics only (no leakage), applied to all
    Fs = np.concatenate([FEAT[s][0] for s in train_subs])
    mu = Fs.mean(0); sd = Fs.std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


class NightDS(torch.utils.data.Dataset):
    def __init__(self, subs, mu, sd):
        self.subs = subs; self.mu = mu; self.sd = sd

    def __len__(self): return len(self.subs)

    def __getitem__(self, i):
        s = self.subs[i]; F, y = FEAT[s]
        return ((F - self.mu) / self.sd).astype(np.float32), y.astype(np.int64), s


def collate(batch):
    lens = [len(b[1]) for b in batch]; mx = max(lens); D = batch[0][0].shape[1]
    X = np.zeros((len(batch), mx, D), np.float32); Y = np.full((len(batch), mx), -100, np.int64)
    M = np.zeros((len(batch), mx), np.float32); ids = []
    for i, (F, y, s) in enumerate(batch):
        X[i, :len(y)] = F; Y[i, :len(y)] = y; M[i, :len(y)] = 1; ids.append(s)
    return torch.tensor(X), torch.tensor(Y), torch.tensor(M), torch.tensor(lens), ids


class FeatSeqNet(nn.Module):
    def __init__(self, in_dim=188, hid=128, layers=2, n_cls=5, dropout=0.4):
        super().__init__()
        self.inln = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(nn.Linear(in_dim, hid), nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(hid, hid, layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(nn.LayerNorm(2 * hid), nn.Dropout(dropout), nn.Linear(2 * hid, n_cls))

    def forward(self, x, lengths):
        h = self.proj(self.inln(x))
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return self.head(out)


def cw(subs, device):
    c = np.zeros(5, np.int64)
    for s in subs: c += np.bincount(FEAT[s][1], minlength=5)
    return torch.tensor(c.sum() / (5 * np.maximum(c, 1)), dtype=torch.float32, device=device)


def metrics(y, p):
    return {"acc": float(accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "kappa": float(cohen_kappa_score(y, p))}


@torch.no_grad()
def subj_probs(model, s, mu, sd, device):
    F, y = FEAT[s]; x = torch.tensor(((F - mu) / sd)[None].astype(np.float32), device=device)
    lo = model(x, torch.tensor([len(y)]))
    return torch.softmax(lo[0].float(), -1).cpu().numpy(), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=10); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-subj", type=int, default=8); ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    subs = list7()
    print(f"FeatSeqNet (BiLSTM over whole-night features) | {len(subs)} subj | {args.folds}-fold")
    for s in tqdm(subs, desc="features", ncols=80): load_feats(s)
    folds = make_folds(subs, args.folds, seed=args.seed); rng = np.random.RandomState(args.seed)
    res, all_probs = [], {}
    for k, (tr_s, te_s) in enumerate(folds):
        vs = sorted(rng.choice(tr_s, size=args.val_subj, replace=False).tolist())
        tc = [s for s in tr_s if s not in vs]
        mu, sd = norm_all(tc)
        dl = torch.utils.data.DataLoader(NightDS(tc, mu, sd), batch_size=args.batch, shuffle=True, collate_fn=collate)
        model = FeatSeqNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        crit = nn.CrossEntropyLoss(weight=cw(tc, device), ignore_index=-100)
        best_val, best = -1, None
        for ep in range(1, args.epochs + 1):
            model.train()
            for X, Y, M, lens, _ in dl:
                X, Y, lens = X.to(device), Y.to(device), lens
                opt.zero_grad(); lo = model(X, lens)
                loss = crit(lo.reshape(-1, 5), Y.reshape(-1)); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            sch.step()
            if ep % 5 == 0 or ep == args.epochs:
                vy = np.concatenate([FEAT[s][1] for s in vs])
                vp = np.concatenate([subj_probs(model, s, mu, sd, device)[0].argmax(1) for s in vs])
                vmf1 = f1_score(vy, vp, average="macro", zero_division=0)
                if vmf1 > best_val: best_val = vmf1; best = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
                if ep % 15 == 0 or ep == args.epochs: print(f"  f{k} ep{ep:02d} val_mF1={vmf1:.3f} (best {best_val:.3f})")
        model.load_state_dict(best)
        ty, pdc = [], []
        for s in te_s:
            pr, yy = subj_probs(model, s, mu, sd, device); all_probs[str(s)] = pr
            ty.append(yy); pdc.append(pr.argmax(1))
        ty = np.concatenate(ty); m = metrics(ty, np.concatenate(pdc)); res.append(m)
        print(f"Fold {k}: FeatSeq acc={m['acc']:.3f} mF1={m['macro_f1']:.3f} k={m['kappa']:.3f}")
    a = np.array([r["acc"] for r in res]); f = np.array([r["macro_f1"] for r in res]); kp = np.array([r["kappa"] for r in res])
    print(f"\n===== FeatSeqNet  acc={a.mean():.4f}+-{a.std():.4f} mF1={f.mean():.4f} kappa={kp.mean():.4f} =====")
    json.dump({"acc": float(a.mean()), "acc_std": float(a.std()), "macro_f1": float(f.mean()), "kappa": float(kp.mean())},
              open(os.path.join(RESULTS, "featseq_all.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(RESULTS, "featseq_probs.npz"), **all_probs)
    print("saved -> results/featseq_all.json, results/featseq_probs.npz")


if __name__ == "__main__":
    main()
