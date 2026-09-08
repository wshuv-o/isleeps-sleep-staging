"""A learned convolutional encoder for the cardiorespiratory branch.

Why this exists. A random 1024-d projection of the raw cardio signal beats the 14
engineered features by +0.097 respiratory AUC, and a randomly initialised MOMENT
reaches 0.7947 against the pretrained 0.8078. So the branch's gain is mostly
capacity over the raw signal, not pretraining, and neither result is an
architectural contribution -- a random feature map is a 2007 technique.

The open question is whether a LEARNED encoder beats the random one. If a small
CNN trained end-to-end on the raw cardio signal exceeds both the engineered
features and the random projection, that is a designed component doing real work,
which is a claim worth making. If it merely matches the random projection, the
honest conclusion is that the engineered features were the bottleneck and no
architecture is needed to say so.

Design follows the cardiorespiratory physiology rather than being copied from an
EEG stager. Respiratory events are slow: a hypopnea unfolds over 10-30 s, the
desaturation lags it by several more. So the kernels are wide and the stack
downsamples aggressively, which is the opposite of the short kernels used for
spindles and K-complexes on the EEG side.

The raw cardio tensor is small enough to keep materialised -- 7 x 750 at 25 Hz is
5250 floats per epoch against 3000 x 7 for EEG -- so this needs no streaming
loader, unlike Arm C.
"""
import numpy as np
import torch
import torch.nn as nn

N_CH, N_T = 7, 750          # 7 cardiorespiratory channels, 30 s at 25 Hz
CHANNELS = ["ECG", "Flow", "Thorax", "Abdomen", "Effort", "SpO2", "Pulse"]


class CardioCNN(nn.Module):
    """[N, 7*750] -> [N, d]. Drop-in replacement for MMFeatureNet.card_enc.

    MMFeatureNet.forward flattens the cardio block before calling card_enc, so
    the module un-flattens here and the surrounding network is untouched.
    """

    def __init__(self, d=64, drop=0.3, width=48):
        super().__init__()
        self.d = d
        # Wide kernels and hard downsampling: respiratory events are slow events,
        # and at 25 Hz a 25-sample kernel is one second.
        self.net = nn.Sequential(
            nn.Conv1d(N_CH, width, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(width), nn.GELU(),
            nn.MaxPool1d(4),                                  # 750 -> ~94
            nn.Dropout(drop),
            nn.Conv1d(width, width * 2, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(width * 2), nn.GELU(),
            nn.MaxPool1d(4),                                  # ~94 -> ~23
            nn.Dropout(drop),
            nn.Conv1d(width * 2, width * 2, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(width * 2), nn.GELU(),
        )
        # Mean and max pooling together: mean carries the epoch's overall level
        # (a sustained desaturation), max carries the transient (an arousal).
        self.head = nn.Sequential(
            nn.Linear(width * 4, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))

    def forward(self, x):
        n = x.shape[0]
        z = x.reshape(n, N_CH, N_T)
        h = self.net(z)
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=-1)
        return self.head(h)


def raw_cardio_table(C, mm_dir):
    """{sid: [n, 7*750] float32} standardised per subject, per channel.

    Matches how load_data z-scores the engineered features, so the only thing
    that differs between this arm and the baseline is what the cardio encoder
    sees, not how it was normalised.
    """
    import glob
    import os
    out = {}
    for f in sorted(glob.glob(os.path.join(mm_dir, "SN*.npz")),
                    key=lambda p: int(os.path.basename(p)[2:-4])):
        sid = int(os.path.basename(f)[2:-4])
        if sid in C.DUP or sid not in C.DATA:
            continue
        card = np.load(f)["card"].astype(np.float32)          # [n, 7, 750]
        mu = card.mean(axis=(0, 2), keepdims=True)
        sd = card.std(axis=(0, 2), keepdims=True) + 1e-6
        card = (card - mu) / sd
        n = len(card)
        if n != len(C.DATA[sid][2]):
            raise ValueError("SN%d: %d cardio epochs vs %d labels"
                             % (sid, n, len(C.DATA[sid][2])))
        out[sid] = card.reshape(n, -1)
    return out
