"""
hag_net_standalone.py -- HAG-Net as a STANDALONE architecture.

The fused version (hag_net.py) takes the gradient-boosting prior's log-probabilities
as an input and mixes them with its own prediction through a router. That makes its
score uninterpretable as an architecture result: it was handed the output of a model
scoring 0.7464 and returned 0.7152.

This version removes the prior entirely. Inputs are only the 188 engineered features
per epoch. It is therefore directly comparable to the other standalone deep models
trained on the same features (FeatSeq BiLSTM: 0.676).

Architecture:
  per-channel MLP encoder (23 -> d) on each of 7 montage channels
       -> graph attention over the montage graph (homologous + neighbour edges)
       -> asymmetry pooling: |z_C4 - z_C3| , |z_O2 - z_O1|      <- the novel prior
       -> fuse [channel mean || asymmetry || event embedding]
       -> bidirectional selective state-space stack over the epoch sequence
       -> per-epoch 5-class logits
"""
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hag_net import GraphAttention, BiSSM, HOMOLOGOUS  # noqa  (reuse the real components)


class HAGNetStandalone(nn.Module):
    def __init__(self, n_base_per_ch=23, n_ch=7, n_event=27, n_cls=5,
                 d=48, D=128, ssm_layers=2, dropout=0.3):
        super().__init__()
        self.n_ch, self.nb = n_ch, n_base_per_ch
        self.enc = nn.Sequential(nn.LayerNorm(n_base_per_ch), nn.Linear(n_base_per_ch, d),
                                 nn.GELU(), nn.Dropout(dropout))
        self.gat = GraphAttention(d)
        self.event_enc = nn.Sequential(nn.LayerNorm(n_event), nn.Linear(n_event, d),
                                       nn.GELU(), nn.Dropout(dropout))
        # [channel mean || asymmetry (2 homologous pairs) || event]
        self.fuse = nn.Sequential(nn.Linear(d + 2 * d + d, D), nn.GELU(), nn.Dropout(dropout))
        self.ssm = nn.ModuleList([BiSSM(D) for _ in range(ssm_layers)])
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Dropout(dropout), nn.Linear(D, n_cls))

    def forward(self, base, event, return_asym=False):
        """base [B,L,7*23]  event [B,L,27]  ->  logits [B,L,5]"""
        B, L, _ = base.shape
        h = self.enc(base.reshape(B * L, self.n_ch, self.nb))       # [B*L, 7, d]
        h = self.gat(h)
        mean = h.mean(1)
        asym = torch.cat([(h[:, i] - h[:, j]).abs() for i, j in HOMOLOGOUS], -1)   # [B*L, 2d]
        ev = self.event_enc(event.reshape(B * L, -1))
        e = self.fuse(torch.cat([mean, asym, ev], -1)).reshape(B, L, -1)
        for blk in self.ssm:
            e = blk(e)
        logits = self.head(e)                                       # no prior, no router
        if return_asym:
            return logits, asym.reshape(B, L, -1).mean(1)
        return logits


# ---- ablation variants: each removes exactly one component -------------------
class NoGraph(HAGNetStandalone):
    """channels mean-pooled instead of attended over the montage graph"""
    def forward(self, base, event, return_asym=False):
        B, L, _ = base.shape
        h = self.enc(base.reshape(B * L, self.n_ch, self.nb))       # graph attention skipped
        mean = h.mean(1)
        asym = torch.cat([(h[:, i] - h[:, j]).abs() for i, j in HOMOLOGOUS], -1)
        ev = self.event_enc(event.reshape(B * L, -1))
        e = self.fuse(torch.cat([mean, asym, ev], -1)).reshape(B, L, -1)
        for blk in self.ssm:
            e = blk(e)
        return self.head(e)


class NoAsym(HAGNetStandalone):
    """asymmetry pooling replaced by zeros: the homologous-pair prior is removed"""
    def forward(self, base, event, return_asym=False):
        B, L, _ = base.shape
        h = self.gat(self.enc(base.reshape(B * L, self.n_ch, self.nb)))
        mean = h.mean(1)
        asym = torch.zeros(B * L, 2 * mean.shape[-1], device=base.device, dtype=mean.dtype)
        ev = self.event_enc(event.reshape(B * L, -1))
        e = self.fuse(torch.cat([mean, asym, ev], -1)).reshape(B, L, -1)
        for blk in self.ssm:
            e = blk(e)
        return self.head(e)


class NoSSM(HAGNetStandalone):
    """temporal model removed: each epoch classified independently"""
    def forward(self, base, event, return_asym=False):
        B, L, _ = base.shape
        h = self.gat(self.enc(base.reshape(B * L, self.n_ch, self.nb)))
        mean = h.mean(1)
        asym = torch.cat([(h[:, i] - h[:, j]).abs() for i, j in HOMOLOGOUS], -1)
        ev = self.event_enc(event.reshape(B * L, -1))
        e = self.fuse(torch.cat([mean, asym, ev], -1)).reshape(B, L, -1)
        return self.head(e)                                          # no BiSSM


VARIANTS = {"HAG-Net (full)": HAGNetStandalone, "- graph attention": NoGraph,
            "- asymmetry pooling": NoAsym, "- SSM (no temporal)": NoSSM}


if __name__ == "__main__":
    b, e = torch.randn(2, 25, 161), torch.randn(2, 25, 27)
    for name, cls in VARIANTS.items():
        m = cls()
        print(f"{name:22s} out={tuple(m(b, e).shape)}  params={sum(p.numel() for p in m.parameters())/1e6:.2f}M")
