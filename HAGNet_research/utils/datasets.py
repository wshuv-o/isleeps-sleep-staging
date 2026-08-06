"""
datasets.py — load preprocessed iSLEEPS .npz and build subject-independent folds.

Conventions:
  - channel order in every npz: ['C4:M1','C3:M2','O2:M1','O1:M2']
  - labels: W0 N1 1 N2 2 N3 3 R4
  - SN28 is a bit-identical duplicate of SN15 -> excluded (see DATA_NOTES.md).
"""
import os
import glob
import numpy as np
from torch.utils.data import Dataset

CHANNELS = ["C4:M1", "C3:M2", "O2:M1", "O1:M2"]
CLASS_NAMES = ["W", "N1", "N2", "N3", "R"]
DUPLICATE_DROP = {28}  # SN28 == SN15

PROC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "processed")


def list_subjects(proc_dir=PROC_DIR, drop_duplicates=True):
    sids = []
    for p in glob.glob(os.path.join(proc_dir, "SN*.npz")):
        sid = int(os.path.basename(p)[2:-4])
        if drop_duplicates and sid in DUPLICATE_DROP:
            continue
        sids.append(sid)
    return sorted(sids)


def load_subject(sid, proc_dir=PROC_DIR, channels=None, normalize=True):
    """Return x [n, C, 3000] float32, y [n] int64 for one subject.
    channels: list of channel names to keep (default all 4). normalize: per-channel
    z-score within this subject (no cross-subject leakage)."""
    d = np.load(os.path.join(proc_dir, f"SN{sid}.npz"), allow_pickle=True)
    x = d["x"].astype(np.float32)            # [n, 4, 3000]
    y = d["y"].astype(np.int64)
    if channels is not None:
        idx = [CHANNELS.index(c) for c in channels]
        x = x[:, idx, :]
    if normalize:
        mu = x.mean(axis=(0, 2), keepdims=True)
        sd = x.std(axis=(0, 2), keepdims=True) + 1e-6
        x = (x - mu) / sd
    return x, y


def make_folds(subjects, n_splits=5, seed=42):
    """Deterministic subject-disjoint k-fold. Returns list of (train_sids, test_sids)."""
    rng = np.random.RandomState(seed)
    sids = list(subjects)
    rng.shuffle(sids)
    folds = [sids[i::n_splits] for i in range(n_splits)]
    out = []
    for k in range(n_splits):
        test = sorted(folds[k])
        train = sorted(s for s in sids if s not in folds[k])
        out.append((train, test))
    return out


class EpochDataset(Dataset):
    """Per-epoch dataset over a set of subjects. x: [C,3000], y: scalar."""

    def __init__(self, subjects, proc_dir=PROC_DIR, channels=None, normalize=True):
        self.channels = channels or CHANNELS
        xs, ys, gs = [], [], []
        for sid in subjects:
            x, y = load_subject(sid, proc_dir, self.channels, normalize)
            xs.append(x); ys.append(y); gs.append(np.full(len(y), sid))
        self.x = np.concatenate(xs, 0)
        self.y = np.concatenate(ys, 0)
        self.groups = np.concatenate(gs, 0)

    def class_counts(self):
        return np.bincount(self.y, minlength=len(CLASS_NAMES))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


if __name__ == "__main__":
    subs = list_subjects()
    print(f"subjects: {len(subs)} (SN28 dropped): {subs}")
    folds = make_folds(subs, n_splits=5)
    for k, (tr, te) in enumerate(folds):
        print(f"  fold {k}: train={len(tr)} test={len(te)} -> test {te}")
    ds = EpochDataset(folds[0][1])  # load test subjects of fold 0
    print("sample x:", ds.x.shape, ds.x.dtype, "y:", ds.y.shape)
    print("class counts:", dict(zip(CLASS_NAMES, ds.class_counts().tolist())))
    print("x mean/std after norm:", round(float(ds.x.mean()), 4), round(float(ds.x.std()), 3))
