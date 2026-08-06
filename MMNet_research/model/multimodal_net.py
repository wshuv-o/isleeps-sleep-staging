"""
multimodal_net.py -- Option A: Two-Stream Cross-Attention, multi-task (stage + apnea).

Every design choice is tied to a property of the data, noted inline. Nothing here is
a trend block dropped in for looks; each piece has a job and is meant to be ablatable.

Data:
    eeg  [B, L, 7, 3000]  @100 Hz  (C4 C3 O2 O1 E1 E2 EMG)  -> staging signal
    card [B, L, 7,  750]  @ 25 Hz  (ECG Flow Thorax Abdomen Effort SpO2 Pulse) -> apnea/arousal
    L = number of consecutive 30 s epochs in the window (sleep is sequential)

Flow:  per-epoch:  EEG-CNN + Cardio-CNN -> cross-modal attention -> fused vector
       per-window: BiLSTM over epochs -> {staging head, apnea head}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cnn_block(ci, co, k, s):
    # conv -> BN -> GELU : BN is important on small data (stabilises, mild regularise)
    return nn.Sequential(nn.Conv1d(ci, co, k, stride=s, padding=k // 2, bias=False),
                         nn.BatchNorm1d(co), nn.GELU())


class EEGEncoder(nn.Module):
    """Multi-scale 1-D CNN. Two branches because sleep EEG carries information at two
    very different time-scales: fast spindles (~12 Hz) and slow delta (~1 Hz). One kernel
    size cannot see both, so we use a small-kernel branch and a large-kernel branch and
    concatenate -- the standard, principled way to read EEG, sized for OUR 100 Hz."""
    def __init__(self, n_ch=7, d=128, drop=0.3):
        super().__init__()
        self.fine = nn.Sequential(                      # small kernels -> spindles, fast graphoelements
            _cnn_block(n_ch, 64, 11, 2), nn.MaxPool1d(4), nn.Dropout(drop),
            _cnn_block(64, 96, 7, 1), nn.MaxPool1d(4))
        self.coarse = nn.Sequential(                    # large kernels -> delta, slow waves
            _cnn_block(n_ch, 64, 51, 6), nn.MaxPool1d(4), nn.Dropout(drop),
            _cnn_block(64, 96, 7, 1), nn.MaxPool1d(2))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Sequential(nn.Linear(96 * 2, d), nn.GELU(), nn.Dropout(drop))

    def forward(self, x):                               # [N, 7, 3000]
        f = self.pool(self.fine(x)).flatten(1)
        c = self.pool(self.coarse(x)).flatten(1)
        return self.out(torch.cat([f, c], -1))          # [N, d]


class CardioEncoder(nn.Module):
    """Lighter CNN. Cardiorespiratory signals are slow and quasi-periodic (heart ~1 Hz,
    breathing ~0.25 Hz, SpO2 slower still), so they carry far less high-frequency detail
    than EEG -- a smaller network is the correct capacity, and over-parameterising here
    would just overfit 96 subjects. Kernels sized (in 25 Hz samples) to span a breath."""
    def __init__(self, n_ch=7, d=64, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            _cnn_block(n_ch, 32, 25, 2), nn.MaxPool1d(4), nn.Dropout(drop),   # ~1 s kernel
            _cnn_block(32, 64, 9, 1), nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1))
        self.out = nn.Sequential(nn.Linear(64, d), nn.GELU(), nn.Dropout(drop))

    def forward(self, x):                               # [N, 7, 750]
        return self.out(self.net(x).flatten(1))         # [N, d]


class CrossModalFusion(nn.Module):
    """The novel core. Treat the per-epoch EEG vector and cardio vector as two tokens and
    let them attend to each other. Motivation specific to THIS cohort: apnea (captured in
    the cardio token) corrupts EEG-based staging, so the model needs a way to let the
    cardio context reshape the EEG representation rather than just concatenating them.
    Ablation: replace this with concatenation -> tests whether the interaction matters."""
    def __init__(self, d_eeg, d_card, D=128, heads=4, drop=0.3):
        super().__init__()
        self.pe = nn.Linear(d_eeg, D)
        self.pc = nn.Linear(d_card, D)
        self.mtype = nn.Parameter(torch.randn(2, D) * 0.02)     # modality-type embeddings
        self.attn = nn.MultiheadAttention(D, heads, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Dropout(drop))

    def forward(self, e, c):                             # [N, d_eeg], [N, d_card]
        tok = torch.stack([self.pe(e), self.pc(c)], 1) + self.mtype[None]   # [N, 2, D]
        a, _ = self.attn(tok, tok, tok)                 # each modality attends to both
        a = self.norm(tok + a)                          # residual
        return self.ff(a.reshape(a.size(0), -1))        # [N, D]  (fuse the two refined tokens)


class MultimodalSleepNet(nn.Module):
    def __init__(self, n_eeg=7, n_card=7, n_cls=5, d_eeg=128, d_card=64, D=128,
                 hidden=128, lstm_layers=2, drop=0.3, fusion="cross"):
        super().__init__()
        self.fusion = fusion
        self.eeg_enc = EEGEncoder(n_eeg, d_eeg, drop)
        self.card_enc = CardioEncoder(n_card, d_card, drop)
        if fusion == "cross":
            self.fuse = CrossModalFusion(d_eeg, d_card, D, drop=drop)
        elif fusion == "concat":                        # ablation: no interaction
            self.fuse = nn.Sequential(nn.Linear(d_eeg + d_card, D), nn.GELU(), nn.Dropout(drop))
        elif fusion == "eeg_only":                      # ablation: drop cardio entirely
            self.fuse = nn.Sequential(nn.Linear(d_eeg, D), nn.GELU(), nn.Dropout(drop))
        # temporal model over epochs: BiLSTM is the data-efficient choice for ~96 subjects
        self.lstm = nn.LSTM(D, hidden, num_layers=lstm_layers, batch_first=True,
                            bidirectional=True, dropout=drop if lstm_layers > 1 else 0.0)
        self.stage_head = nn.Linear(2 * hidden, n_cls)  # 5-class staging
        self.apnea_head = nn.Linear(2 * hidden, 1)      # binary apnea per epoch (multi-task)

    def forward(self, eeg, card):                       # [B,L,7,3000], [B,L,7,750]
        B, L = eeg.shape[:2]
        e = self.eeg_enc(eeg.reshape(B * L, *eeg.shape[2:]))
        c = self.card_enc(card.reshape(B * L, *card.shape[2:]))
        if self.fusion == "cross":
            fz = self.fuse(e, c)
        elif self.fusion == "concat":
            fz = self.fuse(torch.cat([e, c], -1))
        else:
            fz = self.fuse(e)
        h, _ = self.lstm(fz.reshape(B, L, -1))          # [B, L, 2H]
        return self.stage_head(h), self.apnea_head(h).squeeze(-1)   # [B,L,5], [B,L]


if __name__ == "__main__":
    m = MultimodalSleepNet()
    eeg = torch.randn(2, 20, 7, 3000); card = torch.randn(2, 20, 7, 750)
    s, a = m(eeg, card)
    print("stage", tuple(s.shape), "apnea", tuple(a.shape),
          "| params", f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M")
    for name, cls in [("cross", "cross"), ("concat", "concat"), ("eeg_only", "eeg_only")]:
        mm = MultimodalSleepNet(fusion=cls)
        print(f"  fusion={name:8s} params {sum(p.numel() for p in mm.parameters())/1e6:.2f}M")
