"""
deepsleep.py — DeepSleepNet-style stager (stronger than staging_seq).

  DualResCNN   : per-epoch encoder with two parallel branches
                 - small filters (fine temporal / high-freq detail)
                 - large filters (coarse / low-freq, slow waves)
  DeepSleepSeq : encode each epoch -> residual BiLSTM over the sequence -> per-epoch logits
                 (+ a per-epoch head used only for stage-1 encoder pretraining)

Input  : x [B, L, C, 3000] ; Output: logits [B, L, 5].
"""
import torch
import torch.nn as nn


def _bn_relu(c):
    return nn.Sequential(nn.BatchNorm1d(c), nn.ReLU(inplace=True))


class DualResCNN(nn.Module):
    def __init__(self, in_ch=4, fs=100, dropout=0.5, feat_pool=4):
        super().__init__()
        # small-filter branch: fine temporal resolution
        self.small = nn.Sequential(
            nn.Conv1d(in_ch, 64, fs // 2, stride=fs // 16 or 1, padding=fs // 4, bias=False),
            _bn_relu(64), nn.MaxPool1d(8, 8), nn.Dropout(dropout),
            nn.Conv1d(64, 128, 8, padding=4, bias=False), _bn_relu(128),
            nn.Conv1d(128, 128, 8, padding=4, bias=False), _bn_relu(128),
            nn.Conv1d(128, 128, 8, padding=4, bias=False), _bn_relu(128),
            nn.MaxPool1d(4, 4),
        )
        # large-filter branch: coarse / low-frequency structure
        self.large = nn.Sequential(
            nn.Conv1d(in_ch, 64, fs * 4, stride=fs // 2 or 1, padding=fs * 2, bias=False),
            _bn_relu(64), nn.MaxPool1d(4, 4), nn.Dropout(dropout),
            nn.Conv1d(64, 128, 6, padding=3, bias=False), _bn_relu(128),
            nn.Conv1d(128, 128, 6, padding=3, bias=False), _bn_relu(128),
            nn.Conv1d(128, 128, 6, padding=3, bias=False), _bn_relu(128),
            nn.MaxPool1d(2, 2),
        )
        self.gp = nn.AdaptiveAvgPool1d(feat_pool)
        self.feat_dim = 128 * feat_pool * 2

    def forward(self, x):                      # x [B, C, T]
        s = self.gp(self.small(x)).flatten(1)  # [B, 128*pool]
        l = self.gp(self.large(x)).flatten(1)
        return torch.cat([s, l], dim=1)        # [B, feat_dim]


class DeepSleepSeq(nn.Module):
    def __init__(self, in_ch=4, n_classes=5, hidden=256, lstm_layers=2, dropout=0.5):
        super().__init__()
        self.encoder = DualResCNN(in_ch=in_ch, dropout=dropout)
        fd = self.encoder.feat_dim
        self.proj = nn.Sequential(nn.Linear(fd, 256), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.lstm = nn.LSTM(256, hidden, num_layers=lstm_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.res = nn.Linear(256, 2 * hidden)          # residual shortcut (DeepSleepNet)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, n_classes))
        self.epoch_head = nn.Linear(256, n_classes)    # stage-1 pretraining only

    def encode(self, x):                                # x [B, C, T] -> [B, 256]
        return self.proj(self.encoder(x))

    def classify_epoch(self, x):                        # stage-1 per-epoch path
        return self.epoch_head(self.encode(x))

    def forward(self, x):                               # x [B, L, C, T]
        B, L, C, T = x.shape
        f = self.encode(x.reshape(B * L, C, T)).reshape(B, L, -1)  # [B,L,256]
        h, _ = self.lstm(f)                                        # [B,L,2H]
        return self.head(h + self.res(f))                         # residual -> [B,L,5]


if __name__ == "__main__":
    m = DeepSleepSeq(in_ch=4)
    x = torch.randn(2, 20, 4, 3000)
    print("seq out:", m(x).shape)
    print("epoch out:", m.classify_epoch(torch.randn(8, 4, 3000)).shape)
    print("params:", f"{sum(p.numel() for p in m.parameters()):,}",
          "| encoder feat_dim:", m.encoder.feat_dim)
