"""
mm_smoke.py -- proof-of-life / overfit diagnostic on 2 subjects (GPU).

Not a metric (2 subjects can't be). This is an engineering check: a correctly wired
multi-task model MUST be able to overfit a tiny training set. If staging and apnea
training accuracy both climb toward ~1.0, the architecture, gradients, fusion and both
heads are working. If they plateau low, there is a bug -- and we diagnose it, not ditch.

Diagnostics logged every few steps: per-head loss, per-head TRAIN accuracy, and the
staging confusion so we can see WHICH part learns and which doesn't.
"""
import os, sys, glob, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
from multimodal_net import MultimodalSleepNet  # noqa
MM = os.path.join(ROOT, "data", "multimodal")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
L = 20
torch.manual_seed(0); np.random.seed(0)


def pick_two():
    """two subjects with full cardio and all/most stages present."""
    good = []
    for f in sorted(glob.glob(os.path.join(MM, "SN*.npz")), key=lambda p: int(os.path.basename(p)[2:-4])):
        d = np.load(f, allow_pickle=True)
        if int(d["cvalid"].sum()) >= 6 and len(np.unique(d["y"])) >= 4 and d["apnea"].sum() >= 10:
            good.append(os.path.basename(f)[:-4])
        if len(good) == 2:
            break
    return good


def load(sid):
    d = np.load(os.path.join(MM, f"{sid}.npz"), allow_pickle=True)
    eeg = d["eeg"].astype(np.float32); card = d["card"].astype(np.float32)
    # per-channel z-score using this subject's own statistics (SpO2~90, Pulse~78 etc.
    # live on very different scales, so per-channel standardisation is essential)
    eeg = (eeg - eeg.mean((0, 2), keepdims=True)) / (eeg.std((0, 2), keepdims=True) + 1e-6)
    cm = card.mean((0, 2), keepdims=True); cs = card.std((0, 2), keepdims=True) + 1e-6
    card = (card - cm) / cs
    return eeg, card, d["y"].astype(np.int64), d["apnea"].astype(np.int64)


class WinDS(Dataset):
    def __init__(self, subs):
        self.E, self.C, self.Y, self.A, self.idx = {}, {}, {}, {}, []
        for s in subs:
            e, c, y, a = load(s)
            self.E[s], self.C[s], self.Y[s], self.A[s] = e, c, y, a
            for st in range(0, len(y) - L + 1, L // 2):
                self.idx.append((s, st))

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        s, st = self.idx[i]
        return (torch.from_numpy(self.E[s][st:st+L]), torch.from_numpy(self.C[s][st:st+L]),
                torch.from_numpy(self.Y[s][st:st+L]), torch.from_numpy(self.A[s][st:st+L].astype(np.float32)))


def main():
    subs = pick_two()
    print(f"overfit test on {subs} | device {DEV}", flush=True)
    ds = WinDS(subs)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    print(f"{len(ds)} windows of {L} epochs", flush=True)

    model = MultimodalSleepNet(fusion="cross", drop=0.0).to(DEV)   # drop=0: we WANT to overfit
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    ce = nn.CrossEntropyLoss(); bce = nn.BCEWithLogitsLoss()

    t0 = time.time()
    for ep in range(1, 61):
        model.train(); ys, ps, ays, aps = [], [], [], []
        ls, la = 0.0, 0.0
        for eeg, card, y, a in dl:
            eeg, card, y, a = eeg.to(DEV), card.to(DEV), y.to(DEV), a.to(DEV)
            opt.zero_grad()
            s_out, a_out = model(eeg, card)
            loss_s = ce(s_out.reshape(-1, 5), y.reshape(-1))
            loss_a = bce(a_out.reshape(-1), a.reshape(-1))
            loss = loss_s + 0.5 * loss_a
            loss.backward(); opt.step()
            ls += loss_s.item(); la += loss_a.item()
            ys.append(y.reshape(-1).cpu().numpy()); ps.append(s_out.reshape(-1, 5).argmax(-1).cpu().numpy())
            ays.append(a.reshape(-1).cpu().numpy()); aps.append((a_out.reshape(-1) > 0).float().cpu().numpy())
        if ep % 5 == 0 or ep == 1:
            y_, p_ = np.concatenate(ys), np.concatenate(ps)
            ay_, ap_ = np.concatenate(ays), np.concatenate(aps)
            sacc = accuracy_score(y_, p_); smf1 = f1_score(y_, p_, average="macro", zero_division=0)
            aacc = accuracy_score(ay_, ap_)
            print(f"  ep{ep:02d}  stage_loss={ls/len(dl):.3f} apnea_loss={la/len(dl):.3f} | "
                  f"TRAIN stage_acc={sacc:.3f} mF1={smf1:.3f} | apnea_acc={aacc:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    # final verdict + per-class to see which stages learned
    y_, p_ = np.concatenate(ys), np.concatenate(ps)
    pcf = f1_score(y_, p_, average=None, labels=range(5), zero_division=0)
    print("\n  per-class TRAIN F1 (W N1 N2 N3 R):", [round(float(x), 2) for x in pcf])
    ok = accuracy_score(y_, p_) > 0.9
    print("\n  VERDICT:", "PASS - architecture can fit; wiring is correct, go to full run"
          if ok else "FAIL - cannot overfit 2 subjects; there is a bug to diagnose, not ditch")


if __name__ == "__main__":
    main()
