"""
mm_feat_train.py -- train the multimodal feature-fusion net (model/mm_feature_net.py).

Reads data/mm_features (Feeg 188 + Fcard 14 per epoch). Subject-independent split.
Per-subject z-score of both feature sets (removes inter-patient amplitude offsets, the
standard trick that lets a model generalise across people). BiLSTM over L-epoch windows.

Runs the ablation ladder in one go so every component's impact is measured:
    eeg_only : engineered EEG features only          (does cardio add anything?)
    concat   : EEG + cardio, concatenated            (does the modality help at all?)
    cross    : EEG + cardio, cross-modal attention   (does the INTERACTION help?)
Multi-task: staging head + apnea head (pos-weighted BCE) keeps cardio load-bearing.

  KMP_DUPLICATE_LIB_OK=TRUE python extra/mm_feat_train.py
"""
import os, sys, glob, json, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.join(ROOT, "utils"))
from mm_feature_net import MMFeatureNet  # noqa
try:
    from datasets import DUPLICATE_DROP  # noqa
except Exception:
    DUPLICATE_DROP = {28}
FE = os.path.join(ROOT, "data", "mm_features"); RES = os.path.join(ROOT, "results")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, NC = 20, 5
CLS = ["W", "N1", "N2", "N3", "R"]


def load_all(require_cardio=True):
    """{sid: (Feeg_z, Fcard_z, y, apnea, has_card)} with per-subject z-scored features."""
    data = {}
    for f in sorted(glob.glob(os.path.join(FE, "SN*.npz")), key=lambda p: int(os.path.basename(p)[2:-4])):
        sid = int(os.path.basename(f)[2:-4])
        if sid in DUPLICATE_DROP:
            continue
        d = np.load(f)
        Fe = np.nan_to_num(d["Feeg"]).astype(np.float32)
        Fc = np.nan_to_num(d["Fcard"]).astype(np.float32)
        Fe = (Fe - Fe.mean(0)) / (Fe.std(0) + 1e-6)
        Fc = (Fc - Fc.mean(0)) / (Fc.std(0) + 1e-6)
        has = int(d["cvalid"].sum()) >= 5
        data[sid] = (Fe, Fc, d["y"].astype(np.int64), d["apnea"].astype(np.int64), has)
    return data


class WinDS(Dataset):
    def __init__(self, data, subs, stride):
        self.data, self.idx = data, []
        for s in subs:
            n = len(data[s][2])
            for st in range(0, max(1, n - L + 1), stride):
                self.idx.append((s, st))

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        s, st = self.idx[i]; fe, fc, y, a, _ = self.data[s]
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


@torch.no_grad()
def evaluate(model, data, subs):
    model.eval(); ys, ps, ays, apr = [], [], [], []
    for s in subs:
        fe, fc, y, a, _ = data[s]; n = len(y); pad = (-n) % L
        if pad:
            fe = np.concatenate([fe, np.zeros((pad, fe.shape[1]), np.float32)])
            fc = np.concatenate([fc, np.zeros((pad, fc.shape[1]), np.float32)])
        fe = fe.reshape(-1, L, fe.shape[1]); fc = fc.reshape(-1, L, fc.shape[1])
        so, ao = [], []
        for i in range(0, len(fe), 16):
            s_o, a_o = model(torch.from_numpy(fe[i:i+16]).to(DEV), torch.from_numpy(fc[i:i+16]).to(DEV))
            so.append(s_o.softmax(-1).reshape(-1, 5).cpu().numpy())
            ao.append(torch.sigmoid(a_o).reshape(-1).cpu().numpy())
        so = np.concatenate(so)[:n]; ao = np.concatenate(ao)[:n]
        ys.append(y); ps.append(so.argmax(1)); ays.append(a); apr.append(ao)
    y, p = np.concatenate(ys), np.concatenate(ps)
    ay, apf = np.concatenate(ays), np.concatenate(apr)
    auc = roc_auc_score(ay, apf) if len(np.unique(ay)) > 1 else float("nan")
    return dict(acc=accuracy_score(y, p), mf1=f1_score(y, p, average="macro", zero_division=0),
                kappa=cohen_kappa_score(y, p),
                pcf=f1_score(y, p, average=None, labels=range(5), zero_division=0),
                apnea_f1=f1_score(ay, (apf > 0.5).astype(int), zero_division=0), apnea_auc=auc)


def train_one(data, tr, va, te, fusion, epochs=50, patience=8, lr=1e-3, apnea_w=0.5, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    dl = DataLoader(WinDS(data, tr, L // 2), batch_size=32, shuffle=True, drop_last=True)
    model = MMFeatureNet(fusion=fusion).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    cc = np.zeros(5, np.int64)
    for s in tr: cc += np.bincount(data[s][2], minlength=5)
    w = torch.tensor(cc.sum() / (5 * np.maximum(cc, 1)), dtype=torch.float32, device=DEV)
    ce = nn.CrossEntropyLoss(weight=w, reduction="none")
    ac = np.zeros(2, np.int64)
    for s in tr: ac += np.bincount(data[s][3], minlength=2)
    pw = torch.tensor([ac[0] / max(1, ac[1])], dtype=torch.float32, device=DEV)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)

    best, best_state, bad, t0 = -1, None, 0, time.time()
    for ep in range(1, epochs + 1):
        model.train(); tl = 0.0
        for fe, fc, y, apn, m in dl:
            fe, fc, y, apn, m = fe.to(DEV), fc.to(DEV), y.to(DEV), apn.to(DEV), m.to(DEV)
            opt.zero_grad()
            s_o, a_o = model(fe, fc)
            ls = (ce(s_o.reshape(-1, 5), y.reshape(-1)) * m.reshape(-1)).sum() / m.sum().clamp(min=1)
            la = (bce(a_o.reshape(-1), apn.reshape(-1)) * m.reshape(-1)).sum() / m.sum().clamp(min=1)
            loss = ls + apnea_w * la
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tl += loss.item()
        sch.step()
        v = evaluate(model, data, va)
        if v["mf1"] > best:
            best, bad = v["mf1"], 0
            best_state = {k: vv.detach().clone() for k, vv in model.state_dict().items()}
        else:
            bad += 1
        if ep % 2 == 0 or ep == 1:
            print(f"  [{fusion}] ep{ep:02d} loss={tl/len(dl):.3f} | VAL acc={v['acc']:.3f} "
                  f"mF1={v['mf1']:.3f} k={v['kappa']:.3f} pc={[round(float(x),2) for x in v['pcf']]} "
                  f"apneaAUC={v['apnea_auc']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return model, evaluate(model, data, te), best


def main():
    data = load_all()
    subs = sorted(data)
    full = sum(data[s][4] for s in subs)
    rng = np.random.RandomState(42); order = subs[:]; rng.shuffle(order)
    n = len(order); te = order[:n // 5]; va = order[n // 5:n // 5 + max(10, n // 8)]; tr = order[n // 5 + len(va):]
    print(f"{len(subs)} subjects ({full} with cardio) | train {len(tr)} / val {len(va)} / test {len(te)} | {DEV}\n", flush=True)

    results = {}
    for fusion in ["eeg_only", "concat", "cross"]:
        print(f"=== fusion = {fusion} ===", flush=True)
        _, t, bv = train_one(data, tr, va, te, fusion)
        results[fusion] = t
        print(f"  -> TEST acc={t['acc']:.4f} mF1={t['mf1']:.4f} kappa={t['kappa']:.4f} "
              f"per-class={[round(float(x),3) for x in t['pcf']]} | apneaF1={t['apnea_f1']:.3f} "
              f"AUC={t['apnea_auc']:.3f}\n", flush=True)

    print("==== ABLATION (test set) ====")
    for f in ["eeg_only", "concat", "cross"]:
        t = results[f]
        print(f"  {f:9s}  acc={t['acc']:.4f}  mF1={t['mf1']:.4f}  kappa={t['kappa']:.4f}  apneaAUC={t['apnea_auc']:.3f}")
    print("  boosting ensemble (EEG only): 0.7464 / 0.6753 / 0.6415")
    json.dump({f: {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in results[f].items()}
               for f in results}, open(os.path.join(RES, "mm_feat_ablation.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
