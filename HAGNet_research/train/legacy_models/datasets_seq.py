"""
datasets_seq.py — sequence dataset for context-aware sleep staging.

Each item is L consecutive epochs from ONE subject (temporal order preserved):
  x [L, C, 3000], y [L], mask [L]   (mask=0 marks right-padding on the tail seq)

Training uses overlapping sequences (stride < L) for augmentation; evaluation
uses non-overlapping sequences (stride = L) so every real epoch is scored once.
"""
import numpy as np
from torch.utils.data import Dataset
from datasets import load_subject, CHANNELS, CLASS_NAMES  # noqa


def _chunks(n, L, stride):
    if n <= L:
        return [(0, n)]
    starts = list(range(0, n - L + 1, stride))
    if starts[-1] + L < n:           # cover the tail
        starts.append(n - L)
    return [(s, s + L) for s in starts]


class SequenceDataset(Dataset):
    def __init__(self, subjects, seq_len=20, stride=None, channels=None, normalize=True):
        self.L = seq_len
        self.channels = channels or CHANNELS
        stride = stride or seq_len
        self.X, self.Y, self.M = [], [], []
        for sid in subjects:
            x, y = load_subject(sid, channels=self.channels, normalize=normalize)
            n = len(y)
            for a, b in _chunks(n, seq_len, stride):
                xs = x[a:b]; ys = y[a:b]
                m = np.ones(len(ys), np.float32)
                if len(ys) < seq_len:                       # pad tail
                    pad = seq_len - len(ys)
                    xs = np.concatenate([xs, np.zeros((pad,) + xs.shape[1:], np.float32)], 0)
                    ys = np.concatenate([ys, np.zeros(pad, np.int64)], 0)
                    m = np.concatenate([m, np.zeros(pad, np.float32)], 0)
                self.X.append(xs); self.Y.append(ys); self.M.append(m)
        self.X = np.stack(self.X); self.Y = np.stack(self.Y); self.M = np.stack(self.M)

    def class_counts(self):
        valid = self.M.astype(bool)
        return np.bincount(self.Y[valid], minlength=len(CLASS_NAMES))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.M[i]


if __name__ == "__main__":
    from datasets import make_folds, list_subjects
    folds = make_folds(list_subjects(), 5)
    ds = SequenceDataset(folds[0][1], seq_len=20, stride=20)
    print("seqs:", ds.X.shape, "y:", ds.Y.shape, "mask:", ds.M.shape)
    print("valid epochs:", int(ds.M.sum()), "class counts:",
          dict(zip(CLASS_NAMES, ds.class_counts().tolist())))
