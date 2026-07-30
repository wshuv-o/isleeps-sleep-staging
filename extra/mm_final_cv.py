"""
mm_final_cv.py -- the paper's headline table: ONE locked configuration, 10-fold
subject-independent, reporting staging AND apnea from the SAME joint model.

Locked config (justified by the tuning sweeps, mm_stage_tune.py):
    fusion       = cross (two-stream + cross-modal attention)   [ablated vs concat/eeg_only]
    apnea head   = direct cardio pathway, MLP                    [lifts apnea AUC 0.66->0.71]
    class weight = sqrt inverse-frequency                        [acc/mF1 sweet spot]
    context L    = 20 epochs (10 min)                            [beats L=40]
    apnea loss w = 1.0                                           [co-equal task]
    staging      = BiLSTM posteriors + HMM Viterbi smoothing     [+~1pp, removes 1-epoch flips]

Variants (same folds, so differences are attributable to the component):
    eeg_only     : EEG features only
    concat       : EEG + cardio (no attention)
    cross        : EEG + cardio + cross-modal attention   (headline model)
    cross_noflow : cross with airflow zeroed             (apnea non-circularity check)

  KMP_DUPLICATE_LIB_OK=TRUE python extra/mm_final_cv.py
"""
import os, sys, glob, json, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             roc_auc_score, average_precision_score)

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
L, NC, EPS = 20, 5, 1e-12
CLS = ["W", "N1", "N2", "N3", "R"]
FLOW_COLS = [8, 9]


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


class WinDS(Dataset):
    def __init__(self, data, subs, stride, zero_flow=False):
        self.data, self.idx, self.zf = data, [], zero_flow
        for s in subs:
            n = len(data[s][2])
            for st in range(0, max(1, n - L + 1), stride):
                self.idx.append((s, st))

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        s, st = self.idx[i]; fe, fc, y, a = self.data[s]
        fe, fc, yy, aa = fe[st:st+L].copy(), fc[st:st+L].copy(), y[st:st+L], a[st:st+L]
        if self.zf: fc[:, FLOW_COLS] = 0.0
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


def hmm(A_log, pi_log, logp):
    T = logp.shape[0]; dp = np.zeros((T, NC)); bp = np.zeros((T, NC), int); dp[0] = pi_log + logp[0]
    for t in range(1, T):
        sc = dp[t-1][:, None] + A_log; bp[t] = sc.argmax(0); dp[t] = sc.max(0) + logp[t]
    p = np.zeros(T, int); p[-1] = dp[-1].argmax()
    for t in range(T-2, -1, -1): p[t] = bp[t+1, p[t+1]]
    return p


@torch.no_grad()
def infer(model, data, subs, zero_flow=False):
    model.eval(); probs = {}; ays, apr = [], []
    for s in subs:
        fe, fc, y, a = data[s]; n = len(y); pad = (-n) % L
        fc = fc.copy()
        if zero_flow: fc[:, FLOW_COLS] = 0.0
        fe2, fc2 = fe, fc
        if pad:
            fe2 = np.concatenate([fe, np.zeros((pad, fe.shape[1]), np.float32)])
            fc2 = np.concatenate([fc, np.zeros((pad, fc.shape[1]), np.float32)])
        fe2 = fe2.reshape(-1, L, fe.shape[1]); fc2 = fc2.reshape(-1, L, fc.shape[1])
        so, ao = [], []
        for i in range(0, len(fe2), 16):
            s_o, a_o = model(torch.from_numpy(fe2[i:i+16]).to(DEV), torch.from_numpy(fc2[i:i+16]).to(DEV))
            so.append(s_o.softmax(-1).reshape(-1, 5).cpu().numpy())
            ao.append(torch.sigmoid(a_o).reshape(-1).cpu().numpy())
        probs[s] = (np.concatenate(so)[:n], y)
        ays.append(a); apr.append(np.concatenate(ao)[:n])
    return probs, np.concatenate(ays), np.concatenate(apr)


def train_fold(data, tr, va, fusion, zero_flow, epochs=45, patience=8, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    real = "cross" if fusion == "cross_noflow" else fusion
    dl = DataLoader(WinDS(data, tr, L // 2, zero_flow), batch_size=32, shuffle=True, drop_last=True)
    model = MMFeatureNet(fusion=real).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss(weight=sqrt_cw(tr, data), reduction="none")
    ac = np.zeros(2, np.int64)
    for s in tr: ac += np.bincount(data[s][3], minlength=2)
    pw = torch.tensor([ac[0] / max(1, ac[1])], dtype=torch.float32, device=DEV)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
    A = np.ones((NC, NC)); pi = np.ones(NC)
    for s in tr:
        y = data[s][2]; pi[y[0]] += 1
        for x, z in zip(y[:-1], y[1:]): A[x, z] += 1
    A_log = np.log(A / A.sum(1, keepdims=True)); pi_log = np.log(pi / pi.sum())

    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for fe, fc, y, apn, m in dl:
            fe, fc, y, apn, m = fe.to(DEV), fc.to(DEV), y.to(DEV), apn.to(DEV), m.to(DEV)
            opt.zero_grad(); s_o, a_o = model(fe, fc)
            ls = (ce(s_o.reshape(-1, 5), y.reshape(-1)) * m.reshape(-1)).sum() / m.sum().clamp(min=1)
            la = (bce(a_o.reshape(-1), apn.reshape(-1)) * m.reshape(-1)).sum() / m.sum().clamp(min=1)
            (ls + 1.0 * la).backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sch.step()
        pr, _, _ = infer(model, data, va, zero_flow)
        y = np.concatenate([pr[s][1] for s in va]); p = np.concatenate([pr[s][0].argmax(1) for s in va])
        vacc = accuracy_score(y, p)
        if vacc > best:
            best, bad = vacc, 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return model, (A_log, pi_log)


def fold_metrics(model, hmmp, data, te, zero_flow):
    pr, ay, ap = infer(model, data, te, zero_flow)
    A_log, pi_log = hmmp
    yt = np.concatenate([pr[s][1] for s in te])
    praw = np.concatenate([pr[s][0].argmax(1) for s in te])
    phmm = np.concatenate([hmm(A_log, pi_log, np.log(pr[s][0] + EPS)) for s in te])
    auc = roc_auc_score(ay, ap) if len(np.unique(ay)) > 1 else np.nan
    apv = average_precision_score(ay, ap) if len(np.unique(ay)) > 1 else np.nan
    return dict(acc=accuracy_score(yt, praw), mf1=f1_score(yt, praw, average="macro", zero_division=0),
                kappa=cohen_kappa_score(yt, praw),
                acc_h=accuracy_score(yt, phmm), mf1_h=f1_score(yt, phmm, average="macro", zero_division=0),
                kappa_h=cohen_kappa_score(yt, phmm),
                pcf_h=f1_score(yt, phmm, average=None, labels=range(5), zero_division=0).tolist(),
                apnea_auc=auc, apnea_ap=apv, apnea_f1=f1_score(ay, (ap > 0.5).astype(int), zero_division=0))


def main():
    data = load_all(); subs = sorted(data)
    folds = make_folds(subs, 10, seed=42)
    print(f"{len(subs)} subjects | 10-fold subject-independent | locked config | {DEV}\n", flush=True)
    variants = ["eeg_only", "concat", "cross", "cross_noflow"]
    agg = {v: [] for v in variants}
    for fi, (tr_all, te) in enumerate(folds):
        rng = np.random.RandomState(100 + fi); tr_all = list(tr_all); rng.shuffle(tr_all)
        nv = max(10, len(tr_all) // 9); va = tr_all[:nv]; tr = tr_all[nv:]
        for v in variants:
            zf = (v == "cross_noflow"); t0 = time.time()
            model, hmmp = train_fold(data, tr, va, v, zf)
            m = fold_metrics(model, hmmp, data, te, zf)
            agg[v].append(m)
            print(f"fold {fi+1:2d} [{v:12s}] stg raw {m['acc']:.3f}/{m['mf1']:.3f}  +HMM {m['acc_h']:.3f}/{m['mf1_h']:.3f} "
                  f"| apnea AUC {m['apnea_auc']:.3f} AP {m['apnea_ap']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    print("\n==== 10-FOLD MEAN +- STD (locked joint model) ====")
    summ = {}
    for v in variants:
        A = agg[v]
        def ms(k): return float(np.mean([a[k] for a in A])), float(np.std([a[k] for a in A]))
        acc_h, acc_hs = ms("acc_h"); mf1_h, _ = ms("mf1_h"); kap_h, _ = ms("kappa_h")
        auc, aucs = ms("apnea_auc"); apv, _ = ms("apnea_ap")
        pcf = np.mean([a["pcf_h"] for a in A], 0).tolist()
        summ[v] = dict(acc_hmm=acc_h, acc_hmm_std=acc_hs, mf1_hmm=mf1_h, kappa_hmm=kap_h,
                       acc_raw=ms("acc")[0], mf1_raw=ms("mf1")[0],
                       apnea_auc=auc, apnea_auc_std=aucs, apnea_ap=apv, pcf_hmm=pcf)
        print(f"  {v:12s} STAGING(+HMM) acc={acc_h:.4f}+-{acc_hs:.4f} mF1={mf1_h:.4f} k={kap_h:.4f} "
              f"pc={[round(x,2) for x in pcf]} | APNEA AUC={auc:.4f}+-{aucs:.4f} AP={apv:.4f}")
    print("\n  reference -- boosting ensemble (EEG staging): acc 0.7464 / mF1 0.6753 / kappa 0.6415")
    json.dump(summ, open(os.path.join(RES, "mm_final_cv.json"), "w"), indent=2)
    print("saved -> results/mm_final_cv.json")


if __name__ == "__main__":
    main()
