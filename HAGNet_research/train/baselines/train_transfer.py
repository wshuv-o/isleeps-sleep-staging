"""
train_transfer.py — TRANSFER LEARNING: pretrain a sequence stager on healthy Sleep-EDF,
fine-tune on iSLEEPS. Directly attacks the diagnosed failure (a deep net cannot learn good
features from 99 subjects alone) by learning them from a larger external corpus first.

Matched 4-channel montage [central-EEG, occipital-EEG, EOG, EMG]:
  Sleep-EDF: Fpz-Cz, Pz-Oz, EOG horizontal, EMG submental   (data/sleep_edf_full_proc)
  iSLEEPS  : C4:M1,  O2:M1,  E1:M2,          EMG             (data/processed7 cols 0,2,4,6)

Lazy batch-wise loading (all data, no OOM). Saves per-subject probs for stacking.
  KMP_DUPLICATE_LIB_OK=TRUE d:/EEG-TransNet/testenv/python.exe train_transfer.py --folds 10
"""
import os, sys, json, argparse, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "models"))
from datasets import make_folds, DUPLICATE_DROP, CLASS_NAMES  # noqa
from staging_seq import StagingSeqNet  # noqa

PROC7 = os.path.join(HERE, "data", "processed7")
SEDF = os.path.join(HERE, "data", "sleep_edf_full_proc")
RESULTS = os.path.join(HERE, "results")
torch.backends.cudnn.benchmark = True
L = 20
ISL_COLS = [0, 2, 4, 6]     # C4:M1, O2:M1, E1:M2, EMG  from the 7-ch order
DATA = {}                    # key -> (x_f16 [n,4,3000] pre-normalised, y)


def set_seed(s): np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _norm_store(key, xf, y):
    mu = xf.mean((0, 2), keepdims=True); sd = xf.std((0, 2), keepdims=True) + 1e-6
    DATA[key] = (((xf - mu) / sd).astype(np.float16), y.astype(np.int64))


def load_isleeps(sid):
    d = np.load(os.path.join(PROC7, f"SN{sid}.npz"), allow_pickle=True)
    _norm_store(("i", sid), d["x"][:, ISL_COLS, :].astype(np.float32), d["y"])


def load_sedf(rec):
    d = np.load(os.path.join(SEDF, f"{rec}.npz"), allow_pickle=True)
    _norm_store(("s", rec), d["x"].astype(np.float32), d["y"])


def isl_list():
    s = [int(os.path.basename(p)[2:-4]) for p in glob.glob(os.path.join(PROC7, "SN*.npz"))]
    return sorted(x for x in s if x not in DUPLICATE_DROP)


def sedf_list():
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(SEDF, "*.npz")))


def _starts(n, stride):
    if n <= L: return [0]
    st = list(range(0, n - L + 1, stride))
    if st[-1] + L < n: st.append(n - L)
    return st


class LazySeq(Dataset):
    def __init__(self, keys, stride):
        self.index = [(k, a) for k in keys for a in _starts(len(DATA[k][1]), stride)]

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        k, a = self.index[i]; x, y = DATA[k]
        xs = x[a:a + L].astype(np.float32); ys = y[a:a + L].copy(); m = np.ones(len(ys), np.float32)
        if len(ys) < L:
            p = L - len(ys)
            xs = np.concatenate([xs, np.zeros((p, 4, 3000), np.float32)])
            ys = np.concatenate([ys, np.zeros(p, np.int64)]); m = np.concatenate([m, np.zeros(p, np.float32)])
        return xs, ys, m


def cw(keys, device):
    c = np.zeros(5, np.int64)
    for k in keys: c += np.bincount(DATA[k][1], minlength=5)
    return torch.tensor(c.sum() / (5 * np.maximum(c, 1)), dtype=torch.float32, device=device)


def augment(x, noise=0.05, scale=0.15, tmask=0.10):
    B, Ln, C, T = x.shape
    x = x * torch.empty(B, 1, C, 1, device=x.device).uniform_(1 - scale, 1 + scale)
    x = x + torch.randn_like(x) * noise
    w = int(T * tmask)
    if w > 0:
        st = torch.randint(0, T - w + 1, (1,), device=x.device).item(); x[..., st:st + w] = 0.0
    return x


def transition(keys):
    A = np.ones((5, 5)); pi = np.ones(5)
    for k in keys:
        y = DATA[k][1]; pi[y[0]] += 1
        for a, b in zip(y[:-1], y[1:]): A[a, b] += 1
    A /= A.sum(1, keepdims=True); pi /= pi.sum(); return np.log(A + 1e-12), np.log(pi + 1e-12)


def viterbi(le, lA, lpi):
    T, S = le.shape; dp = np.zeros((T, S)); bp = np.zeros((T, S), int); dp[0] = lpi + le[0]
    for t in range(1, T):
        sc = dp[t - 1][:, None] + lA; bp[t] = sc.argmax(0); dp[t] = sc.max(0) + le[t]
    p = np.zeros(T, int); p[-1] = dp[-1].argmax()
    for t in range(T - 2, -1, -1): p[t] = bp[t + 1, p[t + 1]]
    return p


@torch.no_grad()
def subj_probs(model, key, device):
    x, y = DATA[key]; n = len(y); pad = (-n) % L
    xf = x.astype(np.float32)
    if pad: xf = np.concatenate([xf, np.zeros((pad, 4, 3000), np.float32)])
    seqs = xf.reshape(-1, L, 4, 3000); out = []
    for i in range(0, len(seqs), 16):
        xb = torch.tensor(seqs[i:i + 16], device=device)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            lo = model(xb)
        out.append(torch.softmax(lo.float(), -1).reshape(-1, 5).cpu().numpy())
    return np.concatenate(out)[:n], y


def run_epochs(model, dl, opt, crit, scaler, device, epochs, sched=None, val_keys=None, desc=""):
    best_val = -1; best = None
    for ep in range(1, epochs + 1):
        model.train()
        for x, y, m in tqdm(dl, desc=f"{desc} ep{ep:02d}/{epochs}", leave=False, ncols=84, mininterval=1):
            x, y, m = x.to(device), y.to(device), m.to(device); x = augment(x); opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                lo = model(x); loss = crit(lo.reshape(-1, 5), y.reshape(-1))
                loss = (loss * m.reshape(-1)).sum() / m.sum().clamp(min=1)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update()
        if sched: sched.step()
        if val_keys is not None:
            vy = np.concatenate([DATA[k][1] for k in val_keys])
            vp = np.concatenate([subj_probs(model, k, device)[0].argmax(1) for k in val_keys])
            vmf1 = f1_score(vy, vp, average="macro", zero_division=0)
            if vmf1 > best_val: best_val = vmf1; best = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
            print(f"  {desc} ep{ep:02d} val_mF1={vmf1:.3f}{' *' if best_val == vmf1 else ''}")
    return best


def metrics(y, p):
    return {"acc": float(accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "kappa": float(cohen_kappa_score(y, p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=10); ap.add_argument("--pre-epochs", type=int, default=15)
    ap.add_argument("--ft-epochs", type=int, default=25); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--ft-lr", type=float, default=3e-4)
    ap.add_argument("--val-subj", type=int, default=8); ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    isl = isl_list(); sedf = sedf_list()
    print(f"loading iSLEEPS ({len(isl)}) + Sleep-EDF ({len(sedf)}) into RAM...")
    for s in isl: load_isleeps(s)
    for r in sedf: load_sedf(r)
    isl_k = [("i", s) for s in isl]; sedf_k = [("s", r) for r in sedf]

    # ---------- PRETRAIN on Sleep-EDF ----------
    print(f"\n=== PRETRAIN on {len(sedf)} healthy Sleep-EDF recordings ===")
    model = StagingSeqNet(in_ch=4, dropout=0.5).to(device)
    dl = DataLoader(LazySeq(sedf_k, L // 2), batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.pre_epochs)
    crit = nn.CrossEntropyLoss(weight=cw(sedf_k, device), reduction="none")
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    for ep in range(1, args.pre_epochs + 1):
        model.train(); tot = 0; nb = 0
        for x, y, m in tqdm(dl, desc=f"pretrain ep{ep:02d}/{args.pre_epochs}", leave=False, ncols=84, mininterval=1):
            x, y, m = x.to(device), y.to(device), m.to(device); x = augment(x); opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                lo = model(x); loss = crit(lo.reshape(-1, 5), y.reshape(-1))
                loss = (loss * m.reshape(-1)).sum() / m.sum().clamp(min=1)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
        sch.step(); print(f"  pretrain ep{ep:02d} loss={tot/nb:.3f}")
    pre_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(pre_state, os.path.join(RESULTS, "pretrained_sedf.pt"))
    print("pretrained encoder saved.")

    # ---------- FINE-TUNE per fold on iSLEEPS ----------
    folds = make_folds(isl, args.folds, seed=args.seed); rng = np.random.RandomState(args.seed)
    deep_res, hmm_res, all_probs = [], [], {}
    for k, (tr_s, te_s) in enumerate(folds):
        vs = sorted(rng.choice(tr_s, size=args.val_subj, replace=False).tolist())
        tc = [s for s in tr_s if s not in vs]
        tck = [("i", s) for s in tc]; vk = [("i", s) for s in vs]
        model.load_state_dict(pre_state)   # start from pretrained
        dl = DataLoader(LazySeq(tck, L // 2), batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
        opt = torch.optim.Adam(model.parameters(), lr=args.ft_lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.ft_epochs)
        crit = nn.CrossEntropyLoss(weight=cw(tck, device), reduction="none")
        print(f"\n=== Fold {k} | fine-tune train {len(tc)} / val {len(vs)} / test {len(te_s)} ===")
        best = run_epochs(model, dl, opt, crit, scaler, device, args.ft_epochs, sch, vk, desc=f"f{k}")
        model.load_state_dict(best)
        lA, lpi = transition(tck)
        ty, pd_, ph_ = [], [], []
        for s in te_s:
            pr, yy = subj_probs(model, ("i", s), device); all_probs[str(s)] = pr
            ty.append(yy); pd_.append(pr.argmax(1)); ph_.append(viterbi(np.log(pr + 1e-12), lA, lpi))
        ty = np.concatenate(ty); md = metrics(ty, np.concatenate(pd_)); mh = metrics(ty, np.concatenate(ph_))
        deep_res.append(md); hmm_res.append(mh)
        print(f"Fold {k}: transfer acc={md['acc']:.3f} mF1={md['macro_f1']:.3f} | +HMM acc={mh['acc']:.3f} mF1={mh['macro_f1']:.3f} k={mh['kappa']:.3f}")

    def summ(name, res):
        acc = np.array([r["acc"] for r in res]); mf1 = np.array([r["macro_f1"] for r in res]); kap = np.array([r["kappa"] for r in res])
        print(f"{name:14s} acc={acc.mean():.4f}+-{acc.std():.4f} mF1={mf1.mean():.4f} kappa={kap.mean():.4f}")
        return {"acc": float(acc.mean()), "acc_std": float(acc.std()), "macro_f1": float(mf1.mean()), "kappa": float(kap.mean())}
    print("\n===== TRANSFER (pretrain Sleep-EDF -> fine-tune iSLEEPS) =====")
    out = {"transfer": summ("transfer", deep_res), "transfer_hmm": summ("transfer+HMM", hmm_res)}
    json.dump(out, open(os.path.join(RESULTS, "transfer_all.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(RESULTS, "transfer_probs.npz"), **all_probs)
    print("saved -> results/transfer_all.json, results/transfer_probs.npz")


if __name__ == "__main__":
    main()
