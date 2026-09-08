"""Adapt iSLEEPS epochs to CBraMod's input contract.

The four adaptation problems in docs/RTX5080_SETUP_AND_EXPERIMENT.md section 4,
each resolved against CBraMod's own code rather than assumed:

1. Sampling rate. processed7 is 100 Hz; CBraMod patches are 200 samples = 1 s at
   200 Hz. Upsampled with resample_poly (polyphase), not interpolation.
2. Epoch length. 30 s -> exactly 30 one-second patches, which is CBraMod's
   default seq_len=30. No sub-window pooling needed.
3. Montage. processed7 is C4:M1 C3:M2 O2:M1 O1:M2 E1:M2 E2:M2 EMG. CBraMod takes
   channels as a plain dimension, so all 7 pass through; EEG_ONLY selects the
   four EEG leads for checkpoints that reject EOG/EMG.
4. Normalisation. CBraMod's own loaders (isruc, tuab, physio) return seq/100 and
   processed7 is raw uV that was never z-scored, so ONE division by 100 is the
   correct single normalisation -- do not z-score on top of it.

Verified on this data: the released checkpoint loads with zero missing and zero
unexpected keys, and a [32,7,30,200] batch encodes in ~12 ms at 0.14 GB peak.
"""
import numpy as np
import torch
from scipy.signal import resample_poly

SRC_HZ, DST_HZ = 100, 200
PATCH = 200                      # samples per 1 s patch at DST_HZ
SEQ_LEN = 30                     # patches per 30 s epoch == CBraMod default
UV_SCALE = 100.0                 # CBraMod loaders: return seq/100
CHANNELS = ["C4:M1", "C3:M2", "O2:M1", "O1:M2", "E1:M2", "E2:M2", "EMG"]
EEG_ONLY = [0, 1, 2, 3]


def to_cbramod(x, channels=None):
    """processed7 epochs [n, 7, 3000] uV @100 Hz -> [n, C, 30, 200] at CBraMod scale."""
    x = np.asarray(x, dtype=np.float32)
    if channels is not None:
        x = x[:, channels]
    n, c, t = x.shape
    if t != SRC_HZ * SEQ_LEN:
        raise ValueError("expected %d samples per epoch, got %d" % (SRC_HZ * SEQ_LEN, t))
    x = resample_poly(x, DST_HZ // SRC_HZ, 1, axis=-1)
    return x.reshape(n, c, SEQ_LEN, PATCH) / UV_SCALE


def load_encoder(ckpt_path, pretrained=True, device="cuda", seed=42, cbramod_cls=None):
    """CBraMod with proj_out stripped, ready for frozen or fine-tuned use.

    pretrained=False gives the identical architecture at random init -- the Arm D
    control, which is the only way to tell pretraining from architecture.
    """
    if cbramod_cls is None:
        from models.cbramod import CBraMod as cbramod_cls
    torch.manual_seed(seed)
    m = cbramod_cls().to(device)
    if pretrained:
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
        m.load_state_dict(sd, strict=True)      # strict: a silent key mismatch is a bug
    m.proj_out = torch.nn.Identity()
    # eval() matters: CBraMod's encoder has dropout, so a model left in train mode
    # returns a different embedding every call and the frozen cache is silently
    # stochastic. Callers fine-tuning (Arm C) must call .train() themselves.
    m.eval()
    return m


@torch.no_grad()
def embed(model, x, device="cuda", bs=64, channels=None, pool="patch"):
    """Encode processed7 epochs to frozen embeddings.

    pool='patch' -> [n, C, 200], mean over the 30 patches, keeping channels so a
    probe can weight leads differently. Reshape to [n, C*200] for a linear head.
    """
    xr = to_cbramod(x, channels)
    out = []
    for i in range(0, len(xr), bs):
        o = model(torch.tensor(xr[i:i + bs]).to(device))     # [b, C, 30, 200]
        out.append((o.mean(dim=2) if pool == "patch" else o).half().cpu().numpy())
    return np.concatenate(out)
