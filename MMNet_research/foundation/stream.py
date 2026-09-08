"""Streaming raw-signal data path, for fine-tuning the encoder (Arm C).

Arms A, B and D read fixed-width features, so mmnet_core materialises every
overlapping training window and pushes the lot to the GPU in one tensor. That is
~126 MB for the 188-d features. The same tensor for raw 7-channel signal is

    8.4k windows x 20 epochs x 7 ch x 3000 samples x 4 B ~= 14 GB

which does not fit alongside the model on a 16 GB card, so Arm C cannot reuse
that loop. Here the raw signal is held once per subject in float16 (~3.8 GB for
the cohort, on the CPU), windows are indexed on the fly, and only the current
batch is resampled and moved to the GPU.

Resampling 100 -> 200 Hz costs about 0.27 s per batch of 640 epochs on the CPU,
against roughly 0.7 s for the GPU step, so it hides behind the compute when the
loader prefetches.

The encoder is the ONLY thing that differs from arms A/B/D: its output is fed
through the same FeatMLP(-> 128) the feature arms use, then into the unchanged
fusion, BiLSTM, bypass and heads.
"""
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample_poly

import cbramod_adapter as A

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P7 = os.path.join(REPO, "data", "processed7")
EMB = os.path.join(REPO, "data", "cbramod_emb")


class RawStore:
    """Per-subject raw signal at 100 Hz (float16, CPU) plus the aligned labels.

    Deliberately does NOT pre-resample to 200 Hz: that would be 7.6 GB for the
    cohort rather than 3.8 GB, and the resample is cheap enough to do per batch.
    """

    def __init__(self, C, subs=None):
        self.C = C
        self.x, self.fcard, self.y, self.apnea = {}, {}, {}, {}
        for f in sorted(glob.glob(os.path.join(P7, "SN*.npz")),
                        key=lambda p: int(os.path.basename(p)[2:-4])):
            sid = int(os.path.basename(f)[2:-4])
            if sid in C.DUP or sid not in C.DATA or (subs is not None and sid not in subs):
                continue
            d = np.load(f)
            _, fc, y, a = C.DATA[sid]
            if len(d["x"]) != len(y):
                raise ValueError("SN%d: %d raw epochs vs %d in mm_features"
                                 % (sid, len(d["x"]), len(y)))
            if not np.array_equal(d["y"], y):
                raise ValueError("SN%d: label mismatch between processed7 and mm_features" % sid)
            self.x[sid] = d["x"].astype(np.float16)      # [n, 7, 3000]
            self.fcard[sid], self.y[sid], self.apnea[sid] = fc, y, a
        self.subs = sorted(self.x)

    def epochs_to_model(self, raw):
        """[n, 7, 3000] uV @100 Hz -> [n, 7*30*200] at CBraMod scale, flattened.

        Flattened because MMFeatureNet.forward does feeg.reshape(B*Ln, -1) before
        calling eeg_enc; the encoder un-flattens. That keeps MMFeatureNet itself
        untouched.
        """
        z = A.to_cbramod(np.asarray(raw, dtype=np.float32))   # [n, 7, 30, 200], /100 applied
        return z.reshape(len(z), -1)

    def windows(self, subs, stride):
        """Window index as (subject, start) pairs -- the same stride policy as mmnet_core."""
        L = self.C.L
        out = []
        for s in subs:
            n = len(self.y[s])
            for st in range(0, max(1, n - L + 1), stride):
                out.append((s, st))
        return out

    def batch(self, idx, device):
        """Assemble one batch of windows, padding short tails as mmnet_core does."""
        L, C = self.C.L, self.C
        n_card = next(iter(self.fcard.values())).shape[1]
        Fe, Fc, Y, Ap, Mk = [], [], [], [], []
        for s, st in idx:
            raw = self.x[s][st:st + L]
            fc = self.fcard[s][st:st + L]
            y = self.y[s][st:st + L]
            a = self.apnea[s][st:st + L]
            m = np.ones(len(y), np.float32)
            e = self.epochs_to_model(raw)
            if len(y) < L:
                k = L - len(y)
                e = np.concatenate([e, np.zeros((k, e.shape[1]), np.float32)])
                fc = np.concatenate([fc, np.zeros((k, n_card), np.float32)])
                y = np.concatenate([y, np.zeros(k, np.int64)])
                a = np.concatenate([a, np.zeros(k, np.int64)])
                m = np.concatenate([m, np.zeros(k, np.float32)])
            Fe.append(e); Fc.append(fc); Y.append(y); Ap.append(a); Mk.append(m)
        t = lambda arr, d: torch.tensor(np.asarray(arr), dtype=d, device=device)
        return (t(Fe, torch.float32), t(Fc, torch.float32), t(Y, torch.long),
                t(Ap, torch.float32), t(Mk, torch.float32))


class CBraModEncoder(nn.Module):
    """Drop-in replacement for MMFeatureNet.eeg_enc, reading raw signal.

    Input  [N, 7*30*200] (flattened by MMFeatureNet.forward)
    Output [N, d]        same contract as FeatMLP(188, d)

    The trailing FeatMLP is the SAME module arms A/B/D use, so the only
    difference between arms remains what produces its input.

    LayerNorm stands in for the per-subject z-scoring the frozen arms apply to
    cached embeddings: once the encoder is being updated, per-subject statistics
    computed from a frozen pass are stale, so the normalisation has to live
    inside the model.
    """

    def __init__(self, C, ckpt, pretrained=True, d=128, drop=0.3,
                 channels=None, seed=42, trainable=True):
        super().__init__()
        self.n_ch = 7 if channels is None else len(channels)
        self.channels = channels
        self.encoder = A.load_encoder(ckpt, pretrained=pretrained, device="cpu", seed=seed)
        self.pooled_dim = self.n_ch * A.PATCH
        self.norm = nn.LayerNorm(self.pooled_dim)
        self.head = C.FeatMLP(self.pooled_dim, d, drop)
        if not trainable:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.trainable = trainable

    def forward(self, x):
        n = x.shape[0]
        z = x.reshape(n, 7, A.SEQ_LEN, A.PATCH)
        if self.channels is not None:
            z = z[:, self.channels]
        if self.trainable:
            e = self.encoder(z)
        else:
            with torch.no_grad():
                e = self.encoder(z)
        e = e.mean(dim=2).reshape(n, -1)          # pool patches -> [n, n_ch*200]
        return self.head(self.norm(e))

    def train(self, mode=True):
        super().train(mode)
        if not self.trainable:
            self.encoder.eval()               # keep dropout off in the frozen encoder
        return self


def verify_against_cache(C, ckpt, variant="pretrained", n_windows=4, atol=2e-3):
    """Prove the streaming path reproduces the cached embeddings.

    This is the loader's correctness test and it is independent of training: it
    checks resampling, scaling, channel order, window indexing and epoch
    alignment in one go. If this passes, any difference in Arm C's result comes
    from fine-tuning rather than from the data path.
    """
    store = RawStore(C)
    enc = A.load_encoder(ckpt, pretrained=(variant == "pretrained"), device="cuda")
    rng = np.random.RandomState(0)
    worst = 0.0
    for s in rng.choice(store.subs, size=min(n_windows, len(store.subs)), replace=False):
        s = int(s)
        cached = np.load(os.path.join(EMB, variant, "SN%d.npz" % s))["emb"].astype(np.float32)
        raw = store.x[s][:16]
        flat = store.epochs_to_model(raw)
        z = torch.tensor(flat).reshape(len(flat), 7, A.SEQ_LEN, A.PATCH).cuda()
        with torch.no_grad():
            got = enc(z).mean(dim=2).cpu().numpy()      # [16, 7, 200]
        worst = max(worst, float(np.abs(got - cached[:16]).max()))
    ok = worst <= atol
    print("streaming vs cached embeddings: max |diff| = %.2e  -> %s"
          % (worst, "MATCH" if ok else "MISMATCH"))
    return ok
