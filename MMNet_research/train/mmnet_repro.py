"""
mmnet_repro.py -- fast, cache-backed reproduction engine for the MM-Net revision.

Design goals (per the revision brief and the "train once, save on the go" rule):
  * GPU-resident training: all L-epoch windows for a fold live on the GPU, so a 10-fold
    config trains in ~2-3 min instead of ~10. Uses the exact locked configuration that
    produced the paper numbers (cross/concat fusion, sqrt class weights, apnea_w=1, L=20,
    HMM Viterbi smoothing of staging).
  * Every run caches its per-fold and per-subject metrics to results/revision/runs/<name>.json.
    run_config() returns the cache if present, so nothing is ever trained twice.
  * Modality ablation = zeroing the feature columns of the dropped modality (the model
    architecture is unchanged; the information is removed). Masks are derived from the
    feature names, not hard-coded indices.

All results are ten-fold, patient-independent, on the fixed fold assignment make_folds(seed=42).
"""
import os, sys, glob, json
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             roc_auc_score, average_precision_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model")); sys.path.insert(0, os.path.join(ROOT, "utils"))
sys.path.insert(0, os.path.join(ROOT, "processing"))
from mm_feature_net import MMFeatureNet  # noqa
try:
    from datasets import DUPLICATE_DROP, make_folds  # noqa
except Exception:
    DUPLICATE_DROP = {28}
    def make_folds(subs, k, seed=42):
        r = np.random.RandomState(seed); s = list(subs); r.shuffle(s)
        folds = [s[i::k] for i in range(k)]
        return [([x for j, f in enumerate(folds) if j != i for x in f], folds[i]) for i in range(k)]

FE = os.path.join(ROOT, "data", "mm_features")
REV = os.path.join(ROOT, "results", "revision"); RUNS = os.path.join(REV, "runs")
os.makedirs(RUNS, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, NC, EPS = 20, 5, 1e-12
CLS = ["W", "N1", "N2", "N3", "R"]
# cardiorespiratory feature groups (order from cardio_features.cardio_feats)
CARD_GROUPS = {"spo2": [0, 1, 2, 3], "pulse_hrv": [4, 5], "ecg": [6, 7],
               "airflow": [8, 9], "effort": [10, 11, 12, 13]}


def _feature_names():
    from features_v2 import extract_features_v2
    _, names = extract_features_v2(np.zeros((1, 7, 3000), np.float32))
    return names


def eeg_modality_masks():
    """bool masks over the 188 EEG-feature columns for EEG / EOG / EMG."""
    names = _feature_names()
    eeg, eog, emg = np.zeros(188, bool), np.zeros(188, bool), np.zeros(188, bool)
    for i, nm in enumerate(names):
        if any(t in nm for t in ("_c0", "_c1", "_c2", "_c3", "spindle", "sw_")):
            eeg[i] = True
        elif ("_c4" in nm) or ("_c5" in nm) or nm.startswith("eog"):
            eog[i] = True
        elif ("_c6" in nm) or nm.startswith("emg"):
            emg[i] = True
    return {"eeg": eeg, "eog": eog, "emg": emg}


def load_data():
    """{sid:(Feeg[n,188], Fcard[n,14], y[n], apnea[n])} per-subject z-scored; drops SN28."""
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


def _apply_drop(Fe, Fc, eeg_drop, card_drop):
    """return copies with dropped-modality columns zeroed."""
    Fe, Fc = Fe.copy(), Fc.copy()
    em = eeg_modality_masks()
    for m in eeg_drop:
        Fe[:, em[m]] = 0.0
    for g in card_drop:
        if g == "all":
            Fc[:] = 0.0
        else:
            Fc[:, CARD_GROUPS[g]] = 0.0
    return Fe, Fc


def _windows(data, subs, stride, eeg_drop, card_drop):
    """stack L-epoch windows -> GPU tensors Fe,Fc,Y,A,M."""
    FeL, FcL, YL, AL, ML = [], [], [], [], []
    for s in subs:
        Fe, Fc, y, a = data[s]
        Fe, Fc = _apply_drop(Fe, Fc, eeg_drop, card_drop)
        n = len(y)
        for st in range(0, max(1, n - L + 1), stride):
            fe, fc, yy, aa = Fe[st:st+L], Fc[st:st+L], y[st:st+L], a[st:st+L]
            m = np.ones(len(yy), np.float32)
            if len(yy) < L:
                k = L - len(yy)
                fe = np.concatenate([fe, np.zeros((k, 188), np.float32)])
                fc = np.concatenate([fc, np.zeros((k, 14), np.float32)])
                yy = np.concatenate([yy, np.zeros(k, np.int64)]); aa = np.concatenate([aa, np.zeros(k, np.int64)])
                m = np.concatenate([m, np.zeros(k, np.float32)])
            FeL.append(fe); FcL.append(fc); YL.append(yy); AL.append(aa); ML.append(m)
    t = lambda a, d: torch.tensor(np.asarray(a), dtype=d, device=DEV)
    return (t(FeL, torch.float32), t(FcL, torch.float32), t(YL, torch.long),
            t(AL, torch.float32), t(ML, torch.float32))


def _sqrt_cw(data, tr):
    cc = np.zeros(5, np.int64)
    for s in tr: cc += np.bincount(data[s][2], minlength=5)
    inv = np.sqrt(cc.sum() / (5 * np.maximum(cc, 1)))
    return torch.tensor(inv / inv.mean(), dtype=torch.float32, device=DEV)


def _hmm(A_log, pi_log, logp):
    T = logp.shape[0]; dp = np.zeros((T, NC)); bp = np.zeros((T, NC), int); dp[0] = pi_log + logp[0]
    for t in range(1, T):
        sc = dp[t-1][:, None] + A_log; bp[t] = sc.argmax(0); dp[t] = sc.max(0) + logp[t]
    p = np.zeros(T, int); p[-1] = dp[-1].argmax()
    for t in range(T-2, -1, -1): p[t] = bp[t+1, p[t+1]]
    return p


def train_fold(data, tr, va, fusion, eeg_drop, card_drop, epochs=45, patience=8, bs=32, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    Fe, Fc, Y, A, M = _windows(data, tr, L // 2, eeg_drop, card_drop)
    N = Fe.shape[0]
    model = MMFeatureNet(fusion=fusion).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss(weight=_sqrt_cw(data, tr), reduction="none")
    ac = np.zeros(2, np.int64)
    for s in tr: ac += np.bincount(data[s][3], minlength=2)
    pw = torch.tensor([ac[0] / max(1, ac[1])], dtype=torch.float32, device=DEV)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(N, device=DEV)
        for i in range(0, N - bs + 1, bs):
            idx = perm[i:i+bs]
            s_o, a_o = model(Fe[idx], Fc[idx]); m = M[idx].reshape(-1)
            ls = (ce(s_o.reshape(-1, 5), Y[idx].reshape(-1)) * m).sum() / m.sum().clamp(min=1)
            la = (bce(a_o.reshape(-1), A[idx].reshape(-1)) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); (ls + la).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sch.step()
        acc = np.mean([_subj_infer(model, data, s, eeg_drop, card_drop)[0].argmax(1).__eq__(data[s][2]).mean()
                       for s in va])
        if acc > best:
            best, bad = acc, 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state); return model


@torch.no_grad()
def _subj_embed(model, data, s, eeg_drop, card_drop):
    """per-epoch BiLSTM embedding h[n,256] (captured via a forward hook on model.lstm)."""
    model.eval(); Fe, Fc, y, a = data[s]
    Fe, Fc = _apply_drop(Fe, Fc, eeg_drop, card_drop)
    n = len(y); pad = (-n) % L
    fe = np.concatenate([Fe, np.zeros((pad, 188), np.float32)]) if pad else Fe
    fc = np.concatenate([Fc, np.zeros((pad, 14), np.float32)]) if pad else Fc
    fe = torch.tensor(fe.reshape(-1, L, 188), device=DEV); fc = torch.tensor(fc.reshape(-1, L, 14), device=DEV)
    cap = {}
    h = model.lstm.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o[0].detach()))
    out = []
    for i in range(0, len(fe), 32):
        model(fe[i:i+32], fc[i:i+32]); out.append(cap["h"].reshape(-1, cap["h"].shape[-1]).cpu().numpy())
    h.remove()
    return np.concatenate(out)[:n]


@torch.no_grad()
def _subj_infer(model, data, s, eeg_drop, card_drop):
    """return (stage_probs[n,5], apnea_prob[n]) for one subject."""
    model.eval(); Fe, Fc, y, a = data[s]
    Fe, Fc = _apply_drop(Fe, Fc, eeg_drop, card_drop)
    n = len(y); pad = (-n) % L
    fe = np.concatenate([Fe, np.zeros((pad, 188), np.float32)]) if pad else Fe
    fc = np.concatenate([Fc, np.zeros((pad, 14), np.float32)]) if pad else Fc
    fe = torch.tensor(fe.reshape(-1, L, 188), device=DEV); fc = torch.tensor(fc.reshape(-1, L, 14), device=DEV)
    so, ao = [], []
    for i in range(0, len(fe), 32):
        s_o, a_o = model(fe[i:i+32], fc[i:i+32])
        so.append(s_o.softmax(-1).reshape(-1, 5).cpu().numpy()); ao.append(torch.sigmoid(a_o).reshape(-1).cpu().numpy())
    return np.concatenate(so)[:n], np.concatenate(ao)[:n]


def run_config(name, fusion="concat", eeg_drop=(), card_drop=(), force=False, save_embed=False):
    """10-fold; returns dict with per-fold staging(+HMM)/apnea + per-subject staging. Cached.
    save_embed=True also saves (never trains twice): the fold-0 model checkpoint, pooled
    test-set BiLSTM embeddings, and pooled true/predicted labels + apnea scores for figures."""
    cache = os.path.join(RUNS, f"{name}.json")
    if os.path.exists(cache) and not force:
        return json.load(open(cache))
    data = load_data(); subs = sorted(data)
    folds = make_folds(subs, 10, seed=42)
    per_fold, per_subj = [], {}
    E_h, E_y, E_apn, E_sid, P_yt, P_yp, P_ay, P_ap = [], [], [], [], [], [], [], []
    for fi, (tr_all, te) in enumerate(folds):
        rng = np.random.RandomState(100 + fi); tr_all = list(tr_all); rng.shuffle(tr_all)
        nv = max(10, len(tr_all) // 9); va = tr_all[:nv]; tr = tr_all[nv:]
        A = np.ones((NC, NC)); pi = np.ones(NC)
        for s in tr:
            y = data[s][2]; pi[y[0]] += 1
            for x, z in zip(y[:-1], y[1:]): A[x, z] += 1
        A_log = np.log(A / A.sum(1, keepdims=True)); pi_log = np.log(pi / pi.sum())
        model = train_fold(data, tr, va, fusion, list(eeg_drop), list(card_drop))
        if save_embed and fi == 0:
            os.makedirs(os.path.join(REV, "models"), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(REV, "models", f"{name}_fold0.pt"))
        yt, ph, ay, ap = [], [], [], []
        for s in te:
            sp, apn = _subj_infer(model, data, s, list(eeg_drop), list(card_drop))
            y = data[s][2]; pred = _hmm(A_log, pi_log, np.log(sp + EPS))
            per_subj[f"SN{s}"] = dict(acc=float(accuracy_score(y, pred)), kappa=float(cohen_kappa_score(y, pred)),
                                      apnea_scores=apn.tolist())
            yt.append(y); ph.append(pred); ay.append(data[s][3]); ap.append(apn)
            if save_embed:
                h = _subj_embed(model, data, s, list(eeg_drop), list(card_drop))
                E_h.append(h); E_y.append(y); E_apn.append(data[s][3]); E_sid.append(np.full(len(y), s))
        yt, ph = np.concatenate(yt), np.concatenate(ph)
        ay, ap = np.concatenate(ay), np.concatenate(ap)
        if save_embed:
            P_yt.append(yt); P_yp.append(ph); P_ay.append(ay); P_ap.append(ap)
        per_fold.append(dict(
            acc=float(accuracy_score(yt, ph)), mf1=float(f1_score(yt, ph, average="macro", zero_division=0)),
            kappa=float(cohen_kappa_score(yt, ph)),
            pcf=f1_score(yt, ph, average=None, labels=range(5), zero_division=0).tolist(),
            apnea_auc=float(roc_auc_score(ay, ap)) if len(np.unique(ay)) > 1 else float("nan"),
            apnea_ap=float(average_precision_score(ay, ap)) if len(np.unique(ay)) > 1 else float("nan")))
    def agg(k): v = [f[k] for f in per_fold]; return float(np.mean(v)), float(np.std(v))
    out = dict(name=name, fusion=fusion, eeg_drop=list(eeg_drop), card_drop=list(card_drop),
               per_fold=per_fold, per_subject=per_subj,
               acc=agg("acc"), mf1=agg("mf1"), kappa=agg("kappa"),
               apnea_auc=agg("apnea_auc"), apnea_ap=agg("apnea_ap"),
               pcf=np.mean([f["pcf"] for f in per_fold], 0).tolist())
    json.dump(out, open(cache, "w"))
    if save_embed:
        np.savez_compressed(os.path.join(REV, "embeddings.npz"),
                            h=np.concatenate(E_h), stage=np.concatenate(E_y),
                            apnea=np.concatenate(E_apn), sid=np.concatenate(E_sid))
        np.savez_compressed(os.path.join(REV, "predictions.npz"),
                            y_true=np.concatenate(P_yt), y_pred=np.concatenate(P_yp),
                            apnea_true=np.concatenate(P_ay), apnea_score=np.concatenate(P_ap))
        print("saved embeddings.npz, predictions.npz, models/%s_fold0.pt" % name)
    return out


def summary(r):
    return (f"{r['name']:22s} stg acc={r['acc'][0]:.4f}+-{r['acc'][1]:.3f} mF1={r['mf1'][0]:.4f} "
            f"k={r['kappa'][0]:.4f} | apnea AUC={r['apnea_auc'][0]:.4f}+-{r['apnea_auc'][1]:.3f} "
            f"AP={r['apnea_ap'][0]:.4f}")


if __name__ == "__main__":
    r = run_config("headline_concat", fusion="concat")
    print(summary(r))
