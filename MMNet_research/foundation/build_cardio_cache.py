"""Frozen MOMENT embeddings for the CARDIORESPIRATORY stream.

The EEG side gained from importing a representation rather than learning one, and
the same question applies to the other branch -- with better odds, because the
paper's learning curve shows staging has reached this cohort's ceiling while
respiratory detection has not. The head with headroom is the respiratory one.

MOMENT (AutonLab/MOMENT-1-large, 341 M params) is a general time-series
foundation model rather than a physiological one, which is the point: if a
representation learned from arbitrary time series improves respiratory detection,
the mechanism is representation import, not domain-specific pretraining.

Input contract, resolved against the model's config:
  seq_len 512, patch_len 8, flan-t5-large encoder, embedding task.
Our cardio cache is [n, 7, 750] at 25 Hz (30 s), so each epoch is resampled
750 -> 512. MOMENT applies its own instance normalisation, so no scaling is done
here -- that would be the double-normalisation the handoff warns about.

Missing channels matter. 88 of 100 subjects carry all seven cardiorespiratory
channels, 11 carry five or six, and one carries only ECG; absent channels are
stored as exact zeros. Instance-normalising a constant channel divides by ~0, so
those channels are replaced with low-amplitude noise of a scale the model can
normalise without producing NaN, and every output is checked.
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
from scipy.signal import resample_poly

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MM = os.path.join(REPO, "data", "multimodal")
SEQ_LEN = 512
SRC_LEN = 750
CHANNELS = ["ECG", "Flow", "Thorax", "Abdomen", "Effort", "SpO2", "Pulse"]


def encoder(model_id="AutonLab/MOMENT-1-large", device="cuda", pretrained=True, seed=42):
    from momentfm import MOMENTPipeline
    torch.manual_seed(seed)
    m = MOMENTPipeline.from_pretrained(model_id, model_kwargs={"task_name": "embedding"})
    m.init()
    if not pretrained:
        # Arm-D equivalent for this branch: identical architecture, random weights,
        # so a gain can be attributed to pretraining rather than to capacity.
        #
        # Each module is reset with its OWN default initialiser rather than by a
        # blanket rule over parameter shapes. A blanket "xavier for dim>1, zeros
        # otherwise" is wrong and silently catastrophic here: it zeroes LayerNorm
        # gamma, every LayerNorm then outputs exactly zero, and the encoder emits
        # an all-zero embedding. That control would appear to prove pretraining
        # matters while actually comparing against a zero vector.
        n_reset = 0
        for mod in m.modules():
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
                n_reset += 1
        print("random init: reset %d modules with their default initialisers" % n_reset,
              flush=True)
    return m.to(device).eval()


def prepare(card, cvalid, rng):
    """[n,7,750] -> [n,7,512], with absent channels made normalisable."""
    x = resample_poly(np.asarray(card, np.float32), SEQ_LEN, SRC_LEN, axis=-1)
    for c in range(x.shape[1]):
        if not cvalid[c]:
            x[:, c] = rng.normal(0.0, 1e-3, size=x[:, c].shape).astype(np.float32)
    return x


@torch.no_grad()
def embed(model, x, device="cuda", bs=64):
    out = []
    for i in range(0, len(x), bs):
        o = model(x_enc=torch.tensor(x[i:i + bs]).to(device))
        out.append(o.embeddings.float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="moment", choices=["moment", "moment_random"])
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()

    out_dir = os.path.join(REPO, "data", "cardio_emb", a.variant)
    os.makedirs(out_dir, exist_ok=True)
    m = encoder(pretrained=(a.variant == "moment"))
    rng = np.random.RandomState(0)

    files = sorted(glob.glob(os.path.join(MM, "SN*.npz")),
                   key=lambda p: int(os.path.basename(p)[2:-4]))
    t0 = time.time(); done = bad = 0
    for p in files:
        sid = os.path.basename(p)[:-4]
        dst = os.path.join(out_dir, sid + ".npz")
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            done += 1; continue
        d = np.load(p)
        x = prepare(d["card"].astype(np.float32), d["cvalid"], rng)
        e = embed(m, x, bs=a.bs)
        if not np.isfinite(e).all():
            n_bad = int((~np.isfinite(e)).sum())
            print("[BAD] %-6s %d non-finite values -- skipped" % (sid, n_bad), flush=True)
            bad += 1
            continue
        np.savez_compressed(dst, emb=e.astype(np.float16), y=d["y"],
                            cvalid=d["cvalid"], n_valid=int(d["cvalid"].sum()))
        done += 1
        print("[ok] %-6s %5d epochs -> %s  (%d/7 channels)"
              % (sid, len(e), e.shape, int(d["cvalid"].sum())), flush=True)
    print("\n=== %s: %d subjects, %d bad, %.1f min -> %s ==="
          % (a.variant, done, bad, (time.time() - t0) / 60, os.path.relpath(out_dir, REPO)))


if __name__ == "__main__":
    main()
