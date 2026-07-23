"""
staging_seq.py — context-aware sleep stager: per-epoch CNN encoder + BiLSTM.

Input  : x [B, L, C, 3000]   (B sequences of L epochs)
Output : logits [B, L, 5]    (a stage decision for every epoch in the sequence)

The CNN encodes each 30 s epoch to a 128-d vector; a BiLSTM mixes context across
the L epochs so transitions (e.g. N1<->N2, REM onset) use neighbouring epochs —
the standard reason sequence models beat per-epoch models on staging.
"""
import torch
import torch.nn as nn
from staging_cnn import StagingCNN


class StagingSeqNet(nn.Module):
    def __init__(self, in_ch=4, n_classes=5, hidden=128, lstm_layers=2, dropout=0.5):
        super().__init__()
        enc = StagingCNN(in_ch=in_ch, n_classes=n_classes, dropout=dropout)
        self.features = enc.features          # [B,128,T]
        self.pool = nn.AdaptiveAvgPool1d(1)   # -> [B,128,1]
        self.feat_dim = 128
        self.lstm = nn.LSTM(self.feat_dim, hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, n_classes))

    def forward(self, x):
        B, L, C, T = x.shape
        x = x.reshape(B * L, C, T)
        f = self.pool(self.features(x)).flatten(1)   # [B*L, 128]
        f = f.reshape(B, L, self.feat_dim)
        h, _ = self.lstm(f)                          # [B, L, 2*hidden]
        return self.head(h)                          # [B, L, 5]


if __name__ == "__main__":
    m = StagingSeqNet(in_ch=4)
    x = torch.randn(2, 20, 4, 3000)
    out = m(x)
    print("out:", out.shape, "| params:",
          f"{sum(p.numel() for p in m.parameters()):,}")
