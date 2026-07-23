"""
asym_graph_ssm.py -- AsymGraphSSM: a hybrid sleep-staging architecture for stroke.

Novel components, each motivated by the problem (not buzzword salad):
  (1) per-channel CNN encoder                          -> per-channel epoch embeddings
  (2) graph attention over the montage (GAT), with     -> spatial + homologous-hemisphere
      homologous edges C4<->C3, O2<->O1                    message passing
  (3) hemispheric ASYMMETRY pooling (signed diffs)     -> the lesion signal, as an
                                                           explicit inductive bias
  (4) bidirectional selective state-space model        -> long inter-epoch temporal model
      (Mamba-style S6, PURE PYTORCH -- no CUDA kernel)     (Transformer-killer, 2024-25)
  (5) stage head + auxiliary NIHSS (severity) head     -> unifies stager and biomarker

All blocks are pure PyTorch so they run without torch_geometric / mamba_ssm.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 7-channel order: C4:M1(0) C3:M2(1) O2:M1(2) O1:M2(3) E1:M2(4) E2:M2(5) EMG(6)
HOMOLOGOUS = [(0, 1), (2, 3)]          # right<->left central, right<->left occipital
# undirected montage graph (spatial adjacency + homologous + EOG/EMG links)
EDGES = [(0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (0, 4), (1, 5), (2, 4), (3, 5), (6, 4), (6, 5)]


def build_adj(n=7):
    A = torch.eye(n)
    for i, j in EDGES:
        A[i, j] = 1; A[j, i] = 1
    return A


class ChannelCNN(nn.Module):
    """Encode one 30 s channel (3000 samples) -> embedding of size d."""
    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=3, padding=3), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=3, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, d, 5, stride=2, padding=2), nn.BatchNorm1d(d), nn.GELU(),
            nn.AdaptiveAvgPool1d(1))

    def forward(self, x):                      # x [N, 3000]
        return self.net(x.unsqueeze(1)).squeeze(-1)   # [N, d]


class GraphAttention(nn.Module):
    """Single-head graph attention over the fixed channel graph (GAT-style)."""
    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        self.a = nn.Linear(2 * d, 1, bias=False)
        self.register_buffer("adj", build_adj())
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, h):                       # h [B, C, d]
        B, C, d = h.shape
        Wh = self.W(h)                          # [B,C,d]
        hi = Wh.unsqueeze(2).expand(B, C, C, d)
        hj = Wh.unsqueeze(1).expand(B, C, C, d)
        e = self.leaky(self.a(torch.cat([hi, hj], -1)).squeeze(-1))   # [B,C,C]
        mask = self.adj.unsqueeze(0) > 0
        e = e.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(e, dim=-1)        # attention over neighbours
        return F.elu(torch.einsum("bij,bjd->bid", alpha, Wh)) + h     # residual


class AsymPool(nn.Module):
    """Pool channel nodes into one epoch vector, exposing hemispheric asymmetry."""
    def __init__(self, d, out):
        super().__init__()
        # [mean over channels] + |C4-C3| + |O2-O1|  ->  project
        self.proj = nn.Linear(3 * d, out)
        self.asym_dim = 2 * d                   # the raw asymmetry (for the NIHSS head)

    def forward(self, h):                       # h [B, C, d]
        mean = h.mean(1)
        dc = (h[:, 0] - h[:, 1]).abs()          # central asymmetry
        do = (h[:, 2] - h[:, 3]).abs()          # occipital asymmetry
        asym = torch.cat([dc, do], -1)          # [B, 2d]  <- the lesion signal
        e = self.proj(torch.cat([mean, dc, do], -1))
        return e, asym


class SelectiveSSM(nn.Module):
    """Mamba-style selective state-space block (S6), pure-PyTorch sequential scan."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank=None):
        super().__init__()
        self.di = expand * d_model
        self.dt_rank = dt_rank or max(8, d_model // 16)
        self.in_proj = nn.Linear(d_model, 2 * self.di)
        self.conv = nn.Conv1d(self.di, self.di, d_conv, groups=self.di, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.di, self.dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(self.dt_rank, self.di)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                                            .repeat(self.di, 1)))
        self.D = nn.Parameter(torch.ones(self.di))
        self.out_proj = nn.Linear(self.di, d_model)
        self.d_state = d_state

    def forward(self, x):                       # x [B, L, d_model]
        B, L, _ = x.shape
        xz = self.in_proj(x)
        xi, z = xz.chunk(2, dim=-1)             # [B,L,di] each
        xi = self.conv(xi.transpose(1, 2))[..., :L].transpose(1, 2)
        xi = F.silu(xi)
        A = -torch.exp(self.A_log)              # [di, d_state]
        dbl = self.x_proj(xi)                   # [B,L,dt_rank+2*d_state]
        dt, Bm, Cm = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))       # [B,L,di]  (input-dependent step = selective)
        h = x.new_zeros(B, self.di, self.d_state)
        ys = []
        for t in range(L):                      # selective scan (L is the epoch-seq length, small)
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)                 # [B,di,ds]
            dB = dt[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1)        # [B,di,ds]
            h = dA * h + dB * xi[:, t].unsqueeze(-1)
            ys.append(torch.einsum("bds,bs->bd", h, Cm[:, t]))
        y = torch.stack(ys, 1) + self.D * xi    # [B,L,di]
        return self.out_proj(y * F.silu(z))


class BiSSM(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fwd = SelectiveSSM(d); self.bwd = SelectiveSSM(d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        y = self.fwd(x) + self.bwd(x.flip(1)).flip(1)
        return self.norm(x + y)


class AsymGraphSSM(nn.Module):
    def __init__(self, in_ch=7, d=64, D=128, n_classes=5, ssm_layers=2, dropout=0.3):
        super().__init__()
        self.cnn = ChannelCNN(d)
        self.gat = GraphAttention(d)
        self.pool = AsymPool(d, D)
        self.ssm = nn.ModuleList([BiSSM(D) for _ in range(ssm_layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(D, n_classes)
        self.nihss_head = nn.Sequential(nn.Linear(self.pool.asym_dim, 32), nn.GELU(),
                                        nn.Linear(32, 1))       # auxiliary severity head

    def forward(self, x, return_nihss=False):   # x [B, L, C, 3000]
        B, L, C, T = x.shape
        h = self.cnn(x.reshape(B * L * C, T)).reshape(B * L, C, -1)   # [B*L, C, d]
        h = self.gat(h)
        e, asym = self.pool(h)                  # e [B*L, D], asym [B*L, 2d]
        e = e.reshape(B, L, -1)
        for blk in self.ssm:
            e = blk(e)
        logits = self.head(self.drop(e))        # [B, L, n_classes]
        if return_nihss:
            nih = self.nihss_head(asym.reshape(B, L, -1).mean(1)).squeeze(-1)  # [B]
            return logits, nih
        return logits


if __name__ == "__main__":
    m = AsymGraphSSM(in_ch=7)
    x = torch.randn(2, 20, 7, 3000)
    y = m(x)
    n = sum(p.numel() for p in m.parameters())
    print("output", y.shape, "| params", f"{n/1e6:.2f}M")
    lo, nih = m(x, return_nihss=True)
    print("with nihss head:", lo.shape, nih.shape)
