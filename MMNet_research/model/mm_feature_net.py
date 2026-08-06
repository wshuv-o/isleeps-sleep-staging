"""
mm_feature_net.py -- the multimodal thesis model: fuse engineered EEG features with
engineered cardiorespiratory features, then read the whole night with a BiLSTM.

WHY features and not raw signal: on 96 subjects a from-scratch CNN cannot rediscover
what decades of sleep science already put into the 188 EEG features (band power ratios,
spindle/slow-wave detection, Hjorth, EOG/EMG). The raw multimodal net proved this --
it plateaus ~0.62 because it is re-learning features from too few patients. So we hand
the network the strong representations and let it do what deep nets do well: learn the
CROSS-MODAL interaction and the TEMPORAL structure that hand-crafted pipelines cannot.

Architecture (each block ablatable -> each component has a measurable impact):
    Feeg [188] --MLP--> e (128)  \
                                   cross-modal attention --> fused --> BiLSTM --> {stage, apnea}
    Fcard [14] --MLP--> c ( 64)  /
Fusion modes: cross (attention) | concat (interaction removed) | eeg_only (cardio removed).
The multi-task apnea head keeps the cardio stream load-bearing by construction.
"""
import torch
import torch.nn as nn


class FeatMLP(nn.Module):
    """Two-layer MLP with LayerNorm -- turns a raw feature vector into a dense embedding.
    LayerNorm (not BatchNorm) because sequences are short and batches mix subjects."""
    def __init__(self, fin, d, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fin, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))

    def forward(self, x):
        return self.net(x)


class CrossFusion(nn.Module):
    """The novel core. EEG embedding and cardio embedding become two tokens that attend
    to each other. Motivation for THIS cohort (severe apnea, AHI 40-50): respiratory
    events corrupt EEG-based staging and drive cortical arousals, so the model needs the
    cardio context to *reshape* the EEG representation, not just sit beside it.
    Ablation vs 'concat' isolates exactly this interaction's contribution."""
    def __init__(self, d, heads=4, drop=0.3):
        super().__init__()
        self.mtype = nn.Parameter(torch.randn(2, d) * 0.02)      # modality-type embeddings
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(drop))

    def forward(self, e, c):                                      # [N, d], [N, d]
        tok = torch.stack([e, c], 1) + self.mtype[None]          # [N, 2, d]
        a, _ = self.attn(tok, tok, tok)
        a = self.norm(tok + a)                                   # residual
        return self.ff(a.reshape(a.size(0), -1))                 # [N, d]


class MMFeatureNet(nn.Module):
    def __init__(self, n_eeg=188, n_card=14, d=128, d_card=64, hidden=128, layers=2,
                 n_cls=5, drop=0.3, fusion="cross"):
        super().__init__()
        self.fusion = fusion
        self.eeg_enc = FeatMLP(n_eeg, d, drop)
        self.card_enc = FeatMLP(n_card, d_card, drop)
        if fusion == "cross":
            self.card_proj = nn.Linear(d_card, d)                # lift cardio to shared width
            self.fuse = CrossFusion(d, drop=drop)
        elif fusion == "concat":                                 # ablation: no interaction
            self.fuse = nn.Sequential(nn.Linear(d + d_card, d), nn.GELU(), nn.Dropout(drop))
        elif fusion == "eeg_only":                               # ablation: cardio removed
            self.fuse = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(drop))
        # BiLSTM over epochs: sleep is sequential (stage transitions are highly structured);
        # the recurrent model learns this where the ensemble needed a hand-built HMM.
        self.lstm = nn.LSTM(d, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=drop if layers > 1 else 0.0)
        self.stage_head = nn.Linear(2 * hidden, n_cls)
        # Apnea head with a DIRECT cardio pathway: it reads the BiLSTM context AND the raw
        # per-epoch cardio embedding, so SpO2 desaturation / effort reach the respiratory
        # decision without being squeezed through the staging-optimised fusion bottleneck.
        # (eeg_only passes zeros for the cardio part -> the ablation stays truly cardio-free.)
        self.d_card = d_card
        self.uses_cardio = fusion in ("cross", "concat")
        self.apnea_head = nn.Sequential(
            nn.Linear(2 * hidden + d_card, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, 1))

    def forward(self, feeg, fcard):                              # [B, L, 188], [B, L, 14]
        B, Ln = feeg.shape[:2]
        e = self.eeg_enc(feeg.reshape(B * Ln, -1))
        c = self.card_enc(fcard.reshape(B * Ln, -1))
        if self.fusion == "cross":
            fz = self.fuse(e, self.card_proj(c))
        elif self.fusion == "concat":
            fz = self.fuse(torch.cat([e, c], -1))
        else:
            fz = self.fuse(e)
        h, _ = self.lstm(fz.reshape(B, Ln, -1))                  # [B, L, 2H]
        c_seq = c.reshape(B, Ln, -1)
        if not self.uses_cardio:                                 # eeg_only: no cardio leak
            c_seq = torch.zeros_like(c_seq)
        apnea = self.apnea_head(torch.cat([h, c_seq], -1)).squeeze(-1)
        return self.stage_head(h), apnea


if __name__ == "__main__":
    feeg = torch.randn(2, 20, 188); fcard = torch.randn(2, 20, 14)
    for f in ["cross", "concat", "eeg_only"]:
        m = MMFeatureNet(fusion=f)
        s, a = m(feeg, fcard)
        print(f"fusion={f:8s} stage {tuple(s.shape)} apnea {tuple(a.shape)} "
              f"| params {sum(p.numel() for p in m.parameters())/1e6:.3f}M")
