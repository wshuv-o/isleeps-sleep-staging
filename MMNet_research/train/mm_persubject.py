"""
mm_persubject.py -- per-subject staging results (accuracy + kappa) for the proposed
multimodal model and the gradient-boosting ensemble, 10-fold patient-independent.

Each subject is scored once, on the fold where it is in the test set. Both methods use
the same subjects and the same HMM smoothing, so the per-subject columns are comparable.
Saves results/persubject_staging.json = {SNk: {mm_acc, mm_kappa, gb_acc, gb_kappa,
apnea_prev, n_epochs}}.

  KMP_DUPLICATE_LIB_OK=TRUE python extra/mm_persubject.py
"""
import os, sys, glob, json, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model")); sys.path.insert(0, os.path.join(ROOT, "utils"))
from mm_feature_net import MMFeatureNet  # noqa
try:
    from datasets import DUPLICATE_DROP, make_folds  # noqa
except Exception:
    DUPLICATE_DROP = {28}
    def make_folds(subs, k, seed=42):
        r = np.random.RandomState(seed); s = list(subs); r.shuffle(s)
        folds = [s[i::k] for i in range(k)]
        return [([x for j, f in enumerate(folds) if j != i for x in f], folds[i]) for i in range(k)]
FE = os.path.join(ROOT, "data", "mm_features"); RES = os.path.join(ROOT, "results")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, NC, EPS, CTX = 20, 5, 1e-12, 3


def load_all():
    data = {}
    for f in sorted(glob.glob(os.path.join(FE, "SN*.npz")), key=lambda p: int(os.path.basename(p)[2:-4])):
        sid = int(os.path.basename(f)[2:-4])
        if sid in DUPLICATE_DROP:
            continue
        d = np.load(f)
        Fe = np.nan_to_num(d["Feeg"]).astype(np.float32); Fc = np.nan_to_num(d["Fcard"]).astype(np.float32)
        Fe = (Fe - Fe.mean(0)) / (Fe.std(0) + 1e-6); Fc = (Fc - Fc.mean(0)) / (Fc.std(0) + 1e-6)
        data[sid] = (Fe, Fc, d["y"].astype(np.int64), d["apnea"].astype(np.int64))
    return data


def context(F, k=CTX):
    Fp = np.pad(F, ((k, k), (0, 0)), mode="edge")
    return np.concatenate([Fp[i:i + len(F)] for i in range(2 * k + 1)], axis=1)


def hmm(A_log, pi_log, logp):
    T = logp.shape[0]; dp = np.zeros((T, NC)); bp = np.zeros((T, NC), int); dp[0] = pi_log + logp[0]
    for t in range(1, T):
        sc = dp[t-1][:, None] + A_log; bp[t] = sc.argmax(0); dp[t] = sc.max(0) + logp[t]
    p = np.zeros(T, int); p[-1] = dp[-1].argmax()
    for t in range(T-2, -1, -1): p[t] = bp[t+1, p[t+1]]
    return p


class WinDS(Dataset):
    def __init__(self, data, subs, stride):
        self.data, self.idx = data, []
        for s in subs:
            n = len(data[s][2])
            for st in range(0, max(1, n - L + 1), stride):
                self.idx.append((s, st))
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        s, st = self.idx[i]; fe, fc, y, a = self.data[s]
        fe, fc, yy, aa = fe[st:st+L], fc[st:st+L], y[st:st+L], a[st:st+L]
        m = np.ones(len(yy), np.float32)
        if len(yy) < L:
            k = L - len(yy)
            fe = np.concatenate([fe, np.zeros((k, fe.shape[1]), np.float32)])
            fc = np.concatenate([fc, np.zeros((k, fc.shape[1]), np.float32)])
            yy = np.concatenate([yy, np.zeros(k, np.int64)]); aa = np.concatenate([aa, np.zeros(k, np.int64)])
            m = np.concatenate([m, np.zeros(k, np.float32)])
        return (torch.from_numpy(fe), torch.from_numpy(fc), torch.from_numpy(yy),
                torch.from_numpy(aa.astype(np.float32)), torch.from_numpy(m))


def sqrt_cw(tr, data):
    cc = np.zeros(5, np.int64)
    for s in tr: cc += np.bincount(data[s][2], minlength=5)
    inv = np.sqrt(cc.sum() / (5 * np.maximum(cc, 1)))
    return torch.tensor(inv / inv.mean(), dtype=torch.float32, device=DEV)


def train_mmnet(data, tr, va, epochs=45, patience=8):
    torch.manual_seed(42); np.random.seed(42)
    dl = DataLoader(WinDS(data, tr, L // 2), batch_size=32, shuffle=True, drop_last=True)
    model = MMFeatureNet(fusion="cross").to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss(weight=sqrt_cw(tr, data), reduction="none")
    ac = np.zeros(2, np.int64)
    for s in tr: ac += np.bincount(data[s][3], minlength=2)
    pw = torch.tensor([ac[0] / max(1, ac[1])], dtype=torch.float32, device=DEV)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for fe, fc, y, apn, m in dl:
            fe, fc, y, apn, m = fe.to(DEV), fc.to(DEV), y.to(DEV), apn.to(DEV), m.to(DEV)
            opt.zero_grad(); s_o, a_o = model(fe, fc)
            ls = (ce(s_o.reshape(-1, 5), y.reshape(-1)) * m.reshape(-1)).sum() / m.sum().clamp(min=1)
            la = (bce(a_o.reshape(-1), apn.reshape(-1)) * m.reshape(-1)).sum() / m.sum().clamp(min=1)
            (ls + la).backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sch.step()
        # early stop on val accuracy
        acc = np.mean([subj_acc_mm(model, data, s)[0] for s in va])
        if acc > best:
            best, bad = acc, 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience: break
    model.load_state_dict(best_state); return model


@torch.no_grad()
def mm_probs(model, data, s):
    model.eval(); fe, fc, y, a = data[s]; n = len(y); pad = (-n) % L
    fe2 = np.concatenate([fe, np.zeros((pad, fe.shape[1]), np.float32)]) if pad else fe
    fc2 = np.concatenate([fc, np.zeros((pad, fc.shape[1]), np.float32)]) if pad else fc
    fe2 = fe2.reshape(-1, L, fe.shape[1]); fc2 = fc2.reshape(-1, L, fc.shape[1])
    so = []
    for i in range(0, len(fe2), 16):
        s_o, _ = model(torch.from_numpy(fe2[i:i+16]).to(DEV), torch.from_numpy(fc2[i:i+16]).to(DEV))
        so.append(s_o.softmax(-1).reshape(-1, 5).cpu().numpy())
    return np.concatenate(so)[:n], y


def subj_acc_mm(model, data, s):
    p, y = mm_probs(model, data, s); pred = p.argmax(1)
    return accuracy_score(y, pred), y


def main():
    data = load_all(); subs = sorted(data)
    folds = make_folds(subs, 10, seed=42)
    print(f"{len(subs)} subjects | per-subject 10-fold | {DEV}", flush=True)
    per = {}
    for fi, (tr_all, te) in enumerate(folds):
        rng = np.random.RandomState(100 + fi); tr_all = list(tr_all); rng.shuffle(tr_all)
        nv = max(10, len(tr_all) // 9); va = tr_all[:nv]; tr = tr_all[nv:]
        t0 = time.time()
        # HMM transitions from train
        A = np.ones((NC, NC)); pi = np.ones(NC)
        for s in tr:
            y = data[s][2]; pi[y[0]] += 1
            for a, b in zip(y[:-1], y[1:]): A[a, b] += 1
        A_log = np.log(A / A.sum(1, keepdims=True)); pi_log = np.log(pi / pi.sum())
        # gradient-boosting ensemble on EEG features + context
        Xtr = np.concatenate([context(data[s][0]) for s in tr]); ytr = np.concatenate([data[s][2] for s in tr])
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                            class_weight="balanced", random_state=42).fit(Xtr, ytr)
        # multimodal model
        model = train_mmnet(data, tr, va)
        for s in te:
            y = data[s][2]
            mmp, _ = mm_probs(model, data, s); mm_pred = hmm(A_log, pi_log, np.log(mmp + EPS))
            gbp = gb.predict_proba(context(data[s][0])); gb_pred = hmm(A_log, pi_log, np.log(gbp + EPS))
            per[f"SN{s}"] = dict(
                mm_acc=float(accuracy_score(y, mm_pred)), mm_kappa=float(cohen_kappa_score(y, mm_pred)),
                gb_acc=float(accuracy_score(y, gb_pred)), gb_kappa=float(cohen_kappa_score(y, gb_pred)),
                apnea_prev=float(data[s][3].mean()), n_epochs=int(len(y)))
        print(f"fold {fi+1:2d} done ({len(te)} subj, {time.time()-t0:.0f}s)", flush=True)
    json.dump(per, open(os.path.join(RES, "persubject_staging.json"), "w"), indent=2)
    mm_a = np.mean([v["mm_acc"] for v in per.values()]); gb_a = np.mean([v["gb_acc"] for v in per.values()])
    mm_k = np.mean([v["mm_kappa"] for v in per.values()]); gb_k = np.mean([v["gb_kappa"] for v in per.values()])
    print(f"\nsaved {len(per)} subjects -> results/persubject_staging.json")
    print(f"per-subject mean:  MM-Net acc={mm_a:.4f} kappa={mm_k:.4f} | GB acc={gb_a:.4f} kappa={gb_k:.4f}")


if __name__ == "__main__":
    main()
