"""
mm_train.py -- train the multimodal multi-task model on ALL subjects (GPU).

Subject-independent split (train/val/test by patient). Mixed precision for speed on the
6 GB card. Rich diagnostics every epoch so we can DIAGNOSE, not guess:
  - train loss (staging + apnea) to see if it is learning at all
  - val staging acc / macro-F1 / per-class F1  -> which stages work
  - val apnea F1 / AUC                          -> is the cardio branch doing its job
  - train-vs-val gap                            -> overfitting vs underfitting
Early stopping on val staging macro-F1; test reported at the best-val checkpoint.

  KMP_DUPLICATE_LIB_OK=TRUE python extra/mm_train.py --fusion cross
"""
import os, sys, glob, json, time, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
from multimodal_net import MultimodalSleepNet  # noqa
MM = os.path.join(ROOT, "data", "multimodal"); RES = os.path.join(ROOT, "results")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, NC, EPS = 20, 5, 1e-12
CLS = ["W", "N1", "N2", "N3", "R"]
CACHE = {}


def subjects(require_cardio=True):
    out = []
    for f in sorted(glob.glob(os.path.join(MM, "SN*.npz")), key=lambda p: int(os.path.basename(p)[2:-4])):
        d = np.load(f, allow_pickle=True)
        if (not require_cardio) or int(d["cvalid"].sum()) >= 5:
            out.append(os.path.basename(f)[:-4])
    return out


def load(sid):
    if sid in CACHE:
        return CACHE[sid]
    d = np.load(os.path.join(MM, f"{sid}.npz"), allow_pickle=True)
    eeg = d["eeg"].astype(np.float32); card = d["card"].astype(np.float32)
    eeg = (eeg - eeg.mean((0, 2), keepdims=True)) / (eeg.std((0, 2), keepdims=True) + 1e-6)
    card = (card - card.mean((0, 2), keepdims=True)) / (card.std((0, 2), keepdims=True) + 1e-6)
    CACHE[sid] = (eeg, card, d["y"].astype(np.int64), d["apnea"].astype(np.int64))
    return CACHE[sid]


class WinDS(Dataset):
    def __init__(self, subs, stride):
        self.idx = []
        for s in subs:
            n = len(load(s)[2])
            for st in range(0, max(1, n - L + 1), stride):
                self.idx.append((s, st))

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        s, st = self.idx[i]; e, c, y, a = load(s)
        ey = e[st:st+L]; cy = c[st:st+L]; yy = y[st:st+L]; aa = a[st:st+L]
        m = np.ones(len(yy), np.float32)
        if len(yy) < L:
            k = L - len(yy)
            ey = np.concatenate([ey, np.zeros((k, 7, 3000), np.float32)])
            cy = np.concatenate([cy, np.zeros((k, 7, 750), np.float32)])
            yy = np.concatenate([yy, np.zeros(k, np.int64)]); aa = np.concatenate([aa, np.zeros(k, np.int64)])
            m = np.concatenate([m, np.zeros(k, np.float32)])
        return (torch.from_numpy(ey), torch.from_numpy(cy), torch.from_numpy(yy),
                torch.from_numpy(aa.astype(np.float32)), torch.from_numpy(m))


def class_weights(subs):
    c = np.zeros(5, np.int64)
    for s in subs: c += np.bincount(load(s)[2], minlength=5)
    return torch.tensor(c.sum() / (5 * np.maximum(c, 1)), dtype=torch.float32, device=DEV)


@torch.no_grad()
def evaluate(model, subs):
    model.eval(); ys, ps, ays, aps, apr = [], [], [], [], []
    for s in subs:
        e, c, y, a = load(s); n = len(y); pad = (-n) % L
        if pad:
            e = np.concatenate([e, np.zeros((pad, 7, 3000), np.float32)])
            c = np.concatenate([c, np.zeros((pad, 7, 750), np.float32)])
        e = e.reshape(-1, L, 7, 3000); c = c.reshape(-1, L, 7, 750)
        so, ao = [], []
        for i in range(0, len(e), 8):
            with torch.amp.autocast("cuda", enabled=DEV == "cuda"):
                s_o, a_o = model(torch.from_numpy(e[i:i+8]).to(DEV), torch.from_numpy(c[i:i+8]).to(DEV))
            so.append(s_o.float().softmax(-1).reshape(-1, 5).cpu().numpy())
            ao.append(torch.sigmoid(a_o.float()).reshape(-1).cpu().numpy())
        so = np.concatenate(so)[:n]; ao = np.concatenate(ao)[:n]
        ys.append(y); ps.append(so.argmax(1)); ays.append(a); aps.append((ao > 0.5).astype(int)); apr.append(ao)
    y, p = np.concatenate(ys), np.concatenate(ps)
    ay, ap, apf = np.concatenate(ays), np.concatenate(aps), np.concatenate(apr)
    auc = roc_auc_score(ay, apf) if len(np.unique(ay)) > 1 else float("nan")
    return dict(acc=accuracy_score(y, p), mf1=f1_score(y, p, average="macro", zero_division=0),
                kappa=cohen_kappa_score(y, p),
                pcf=f1_score(y, p, average=None, labels=range(5), zero_division=0),
                apnea_f1=f1_score(ay, ap, zero_division=0), apnea_auc=auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion", default="cross", choices=["cross", "concat", "eeg_only"])
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--apnea-w", type=float, default=0.5); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    subs = subjects(require_cardio=(a.fusion != "eeg_only"))
    rng = np.random.RandomState(a.seed); rng.shuffle(subs)
    n = len(subs); te = subs[:n//5]; va = subs[n//5:n//5+max(8, n//8)]; tr = subs[n//5+len(va):]
    print(f"fusion={a.fusion} | {n} subjects -> train {len(tr)} / val {len(va)} / test {len(te)} | {DEV}", flush=True)

    dl = DataLoader(WinDS(tr, L // 2), batch_size=a.batch, shuffle=True, num_workers=0,
                    pin_memory=True, drop_last=True)
    model = MultimodalSleepNet(fusion=a.fusion).to(DEV)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M | {len(dl)} steps/epoch", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    ce = nn.CrossEntropyLoss(weight=class_weights(tr), reduction="none")
    # apnea is ~16% positive; plain BCE collapses to all-negative (apnea_F1=0). Weight the
    # positive class by neg/pos so the apnea head actually predicts events.
    apn_c = np.zeros(2, np.int64)
    for s in tr: apn_c += np.bincount(load(s)[3], minlength=2)
    pw = torch.tensor([apn_c[0] / max(1, apn_c[1])], dtype=torch.float32, device=DEV)
    print(f"apnea pos_weight={pw.item():.2f} (neg/pos)", flush=True)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")

    best, best_state, bad = -1, None, 0; t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); tl_s = tl_a = 0.0
        for eeg, card, y, apn, m in dl:
            eeg, card, y, apn, m = eeg.to(DEV), card.to(DEV), y.to(DEV), apn.to(DEV), m.to(DEV)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=DEV == "cuda"):
                s_o, a_o = model(eeg, card)
                ls = (ce(s_o.reshape(-1, 5), y.reshape(-1)) * m.reshape(-1)).sum() / m.sum().clamp(min=1)
                la = (bce(a_o.reshape(-1), apn.reshape(-1)) * m.reshape(-1)).sum() / m.sum().clamp(min=1)
                loss = ls + a.apnea_w * la
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update()
            tl_s += ls.item(); tl_a += la.item()
        sch.step()
        v = evaluate(model, va)
        if v["mf1"] > best:
            best, bad = v["mf1"], 0
            best_state = {k: vv.detach().clone() for k, vv in model.state_dict().items()}
        else:
            bad += 1
        print(f"ep{ep:02d} loss_s={tl_s/len(dl):.3f} loss_a={tl_a/len(dl):.3f} | "
              f"VAL acc={v['acc']:.3f} mF1={v['mf1']:.3f} k={v['kappa']:.3f} "
              f"pc={[round(float(x),2) for x in v['pcf']]} | apneaF1={v['apnea_f1']:.3f} "
              f"AUC={v['apnea_auc']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if bad >= a.patience:
            print(f"early stop (best val mF1 {best:.4f})", flush=True); break

    model.load_state_dict(best_state)
    t = evaluate(model, te)
    print(f"\n==== TEST (best-val ckpt) | fusion={a.fusion} ====")
    print(f"  STAGING  acc={t['acc']:.4f} mF1={t['mf1']:.4f} kappa={t['kappa']:.4f} "
          f"per-class={[round(float(x),3) for x in t['pcf']]}")
    print(f"  APNEA    F1={t['apnea_f1']:.4f} AUC={t['apnea_auc']:.4f}")
    print(f"  vs boosting ensemble staging: 0.7464 / 0.6753 / 0.6415")
    json.dump({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in t.items()},
              open(os.path.join(RES, f"mm_{a.fusion}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
