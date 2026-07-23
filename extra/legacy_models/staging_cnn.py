"""
staging_cnn.py — compact per-epoch CNN for sleep staging (TinySleepNet-style
representation learner). Input [B, C, 3000] @100 Hz, output 5-class logits.

Intentionally an established, small backbone: the project's contribution is the
stroke-population finding + lesion-aware analysis, not a new architecture.
"""
import torch
import torch.nn as nn


class StagingCNN(nn.Module):
    def __init__(self, in_ch=4, n_classes=5, fs=100, dropout=0.5):
        super().__init__()
        k1 = fs // 2          # 50
        s1 = fs // 16 or 1    # ~6
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 128, k1, stride=s1, padding=k1 // 2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(8, 8), nn.Dropout(dropout),

            nn.Conv1d(128, 128, 8, padding=4, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 8, padding=4, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 8, padding=4, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(4, 4), nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = StagingCNN(in_ch=4)
    x = torch.randn(2, 4, 3000)
    print("out:", m(x).shape, "| params:", f"{count_params(m):,}")
