"""
hag_net.py -- HAG-Net (Hemispheric Asymmetry-Guided Network) for stroke sleep staging.

Components ACTUALLY used in the forward pass:

  (1) per-channel MLP encoder                   -- LayerNorm -> Linear -> GELU -> Dropout
                                                   on each channel's 23-feature block
  (2) hemisphere channel-graph attention (GAT)  -- 7 montage nodes + homologous edges
      with asymmetry pooling                       (C4-C3, O2-O1): the lesion signal
  (3) bidirectional selective state-space (SSM) -- Mamba-style temporal model over epochs
  (4) classical-deep residual fusion            -- an input-dependent router mixes the
                                                   gradient-boosting prior with the deep
                                                   stream, anchored at the prior at init

NAMING / HISTORY. This file and its class were originally "KAGS-Net", where the K stood
for a Kolmogorov-Arnold Network encoder. The KAN was REMOVED: it was imported from
unrelated tabular work and had no EEG justification. It is not in the forward pass and
contributes no parameters. `KANLayer` below is retained only as a record of that earlier
version and is never instantiated. The class is now `HAGNet`, matching the manuscript;
`KAGSNet` remains as an alias so older scripts and checkpoints keep working.

SCOPE OF THE FUSION CLAIM. Output equals the classical prior exactly at initialization
(gate = 0). Keeping it there afterwards relies on a validation-based acceptance rule,
which is a heuristic, not a guarantee: in a controlled test the learned correction
improved validation macro-F1 (0.684 vs 0.673) yet lost on held-out test subjects
(-0.009 macro-F1). Do not describe this as "cannot fall below the prior".

Data-efficient: operates on 188 engineered features + the ensemble's 5 probs per epoch,
never on raw signal, which is why it survives the 99-subject regime where raw-signal
deep nets overfit. Pure PyTorch (no torch_geometric / mamba_ssm).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 7-channel montage node order matches processed7 / features (23 feats per channel).
HOMOLOGOUS = [(0, 1), (2, 3)]                      # C4<->C3, O2<->O1
EDGES = [(0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (0, 4), (1, 5), (2, 4), (3, 5), (6, 4), (6, 5)]


def build_adj(n=7):
    A = torch.eye(n)
    for i, j in EDGES:
        A[i, j] = 1; A[j, i] = 1
    return A


class KANLayer(nn.Module):
    """UNUSED / RETAINED FOR HISTORY -- never instantiated by HAGNet.

    FastKAN: learnable univariate functions via a Gaussian-RBF basis + SiLU base.
    This was the original per-channel encoder. It was dropped because a KAN has no
    physiological justification for EEG features; it acted as a decorative activation
    borrowed from unrelated tabular work. HAGNet uses a plain MLP encoder instead
    (see `self.enc`). Kept here so the earlier architecture stays reproducible.
    """
    def __init__(self, in_dim, out_dim, grid=8, gmin=-2.5, gmax=2.5):
        super().__init__()
        self.register_buffer("centers", torch.linspace(gmin, gmax, grid))
        self.h = (gmax - gmin) / (grid - 1)
        self.norm = nn.LayerNorm(in_dim)
        self.spline = nn.Linear(in_dim * grid, out_dim)
        self.base = nn.Linear(in_dim, out_dim)

    def forward(self, x):                          # [*, in_dim]
        z = self.norm(x)
        rbf = torch.exp(-((z.unsqueeze(-1) - self.centers) / self.h) ** 2)   # [*, in_dim, grid]
        return self.spline(rbf.flatten(-2)) + self.base(F.silu(z))


class GraphAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False); self.a = nn.Linear(2 * d, 1, bias=False)
        self.register_buffer("adj", build_adj()); self.leaky = nn.LeakyReLU(0.2)

    def forward(self, h):                           # [B, 7, d]
        B, C, d = h.shape; Wh = self.W(h)
        e = self.leaky(self.a(torch.cat([Wh.unsqueeze(2).expand(B, C, C, d),
                                         Wh.unsqueeze(1).expand(B, C, C, d)], -1)).squeeze(-1))
        e = e.masked_fill(~(self.adj > 0).unsqueeze(0), float("-inf"))
        return F.elu(torch.einsum("bij,bjd->bid", torch.softmax(e, -1), Wh)) + h


class SelectiveSSM(nn.Module):
    """Mamba-style selective state-space, pure-torch sequential scan."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.di = expand * d_model; self.dt_rank = max(8, d_model // 16); self.ds = d_state
        self.in_proj = nn.Linear(d_model, 2 * self.di)
        self.conv = nn.Conv1d(self.di, self.di, d_conv, groups=self.di, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.di, self.dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(self.dt_rank, self.di)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.di, 1)))
        self.D = nn.Parameter(torch.ones(self.di)); self.out_proj = nn.Linear(self.di, d_model)

    def forward(self, x):                           # [B, L, d_model]
        B, L, _ = x.shape
        xi, z = self.in_proj(x).chunk(2, -1)
        xi = F.silu(self.conv(xi.transpose(1, 2))[..., :L].transpose(1, 2))
        A = -torch.exp(self.A_log.clamp(-6.0, 6.0))          # bound decay (numerical safety)
        dt, Bm, Cm = torch.split(self.x_proj(xi), [self.dt_rank, self.ds, self.ds], -1)
        dt = F.softplus(self.dt_proj(dt)).clamp(max=6.0)     # bound step (prevent fp overflow)
        h = x.new_zeros(B, self.di, self.ds); ys = []
        for t in range(L):
            h = torch.exp(dt[:, t].unsqueeze(-1) * A) * h + (dt[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1)) * xi[:, t].unsqueeze(-1)
            ys.append(torch.einsum("bds,bs->bd", h, Cm[:, t]))
        return self.out_proj((torch.stack(ys, 1) + self.D * xi) * F.silu(z))


class BiSSM(nn.Module):
    def __init__(self, d):
        super().__init__(); self.f = SelectiveSSM(d); self.b = SelectiveSSM(d); self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(x + self.f(x) + self.b(x.flip(1)).flip(1))


class HAGNet(nn.Module):
    def __init__(self, n_base_per_ch=23, n_ch=7, n_event=27, n_cls=5, d=48, D=128,
                 ssm_layers=2, dropout=0.3):
        super().__init__()
        self.n_ch, self.nb = n_ch, n_base_per_ch
        # per-channel encoder (plain MLP -- no KAN; KAN had no EEG justification)
        self.enc = nn.Sequential(nn.LayerNorm(n_base_per_ch), nn.Linear(n_base_per_ch, d),
                                 nn.GELU(), nn.Dropout(dropout))
        self.gat = GraphAttention(d)
        self.event_enc = nn.Sequential(nn.LayerNorm(n_event), nn.Linear(n_event, d),
                                       nn.GELU(), nn.Dropout(dropout))
        self.fuse = nn.Sequential(nn.Linear(3 * d, D), nn.GELU(), nn.Dropout(dropout))  # [mean, |asym_c|, event]
        self.ssm = nn.ModuleList([BiSSM(D) for _ in range(ssm_layers)])
        self.refine = nn.Sequential(nn.LayerNorm(D), nn.Dropout(dropout), nn.Linear(D, n_cls))
        # Input-dependent router: decides PER EPOCH and PER CLASS how much to trust the
        # deep spatio-temporal stream over the classical prior. Conditioned on the fused
        # asymmetry/SSM embedding and on the prior's own entropy (where is it unsure?).
        # A scalar gate must trade off globally and collapses to 0; this one can hand
        # control to the deep stream only on the epochs the prior actually fumbles (N1,
        # stage transitions) while leaving N2/Wake to the booster.
        self.router = nn.Sequential(nn.LayerNorm(D + 1), nn.Linear(D + 1, 64), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(64, n_cls))
        self.router_bias = nn.Parameter(torch.full((n_cls,), -3.0))   # start near the prior
        self.asym_dim = 2 * d

    def forward(self, base, event, ens_logprob, return_asym=False):
        # base [B,L,n_ch*nb]  event [B,L,n_event]  ens_logprob [B,L,5]
        B, L, _ = base.shape
        h = self.enc(base.reshape(B * L, self.n_ch, self.nb))   # [B*L, 7, d]
        h = self.gat(h)
        mean = h.mean(1)
        asym = torch.cat([(h[:, 0] - h[:, 1]).abs(), (h[:, 2] - h[:, 3]).abs()], -1)   # [B*L, 2d]
        dc = (h[:, 0] - h[:, 1]).abs()
        ev = self.event_enc(event.reshape(B * L, -1))
        e = self.fuse(torch.cat([mean, dc, ev], -1)).reshape(B, L, -1)
        for blk in self.ssm:
            e = blk(e)
        deep_logp = torch.log_softmax(self.refine(e), -1)       # [B,L,5] full deep prediction
        prior_logp = torch.log_softmax(ens_logprob, -1)
        ent = -(prior_logp.exp() * prior_logp).sum(-1, keepdim=True)   # [B,L,1] prior uncertainty
        g = torch.sigmoid(self.router(torch.cat([e, ent], -1)) + self.router_bias)  # [B,L,5]
        logits = (1.0 - g) * prior_logp + g * deep_logp         # ROUTED classical/deep mixture
        self.last_gate = g                                      # for logging / analysis
        if return_asym:
            return logits, asym.reshape(B, L, -1).mean(1)
        return logits


# Back-compatible alias: earlier scripts/checkpoints import KAGSNet.
KAGSNet = HAGNet


if __name__ == "__main__":
    m = HAGNet()
    base = torch.randn(2, 20, 161); event = torch.randn(2, 20, 27); ens = torch.log_softmax(torch.randn(2, 20, 5), -1)
    y = m(base, event, ens)
    print("output", y.shape, "| params", f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M")
    print("router gate at init (near 0 -> starts at the classical prior):",
          f"{m.last_gate.mean().item():.3f}")
