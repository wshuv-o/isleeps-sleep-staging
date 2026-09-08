"""Precompute frozen LaBraM embeddings -- a second, independent foundation encoder.

Why a second one. CBraMod's embedding turned out to be complementary to the 188
engineered features (their union beats both). That could be a property of
foundation embeddings in general, or a property of one checkpoint. A second
encoder with different pretraining and different tokenisation distinguishes the
two, and if it also contributes then the multi-representation architecture rests
on a mechanism rather than on one lucky model.

Adaptation, resolved against the model's own contract rather than assumed:

  * Montage. LaBraM demands canonical 10-20 names and rejects our mastoid
    references outright, so C4:M1 C3:M2 O2:M1 O1:M2 -> C4 C3 O2 O1. EOG and EMG
    have no canonical EEG name and are dropped, so LaBraM sees strictly LESS
    signal than CBraMod, which took all seven channels.
  * Sampling rate and window. n_times is fixed at 3000 and the positional
    embedding rejects anything longer, so a 30 s epoch cannot be fed whole at
    LaBraM's native 200 Hz. Feeding our native 100 Hz "works" silently but halves
    every frequency -- a 10 Hz spindle would present as 5 Hz. Instead: resample to
    200 Hz, split the epoch into two 15 s sub-windows of 3000 samples each, encode
    both and mean-pool, which is the sub-window pooling the handoff anticipated.
  * Scale. uV/100, matching the convention used for CBraMod.

Verified before caching: the two sub-window embeddings differ, so pooling carries
information rather than averaging duplicates.
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
from scipy.signal import resample_poly

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P7 = os.path.join(REPO, "data", "processed7")

EEG_IDX = [0, 1, 2, 3]                       # C4:M1 C3:M2 O2:M1 O1:M2
CH_NAMES = ["C4", "C3", "O2", "O1"]          # canonical 10-20 equivalents
SRC_HZ, DST_HZ = 100, 200
SUB_SAMPLES = 3000                           # LaBraM n_times (15 s at 200 Hz)
UV_SCALE = 100.0


def encoder(pretrained=True, device="cuda", seed=42):
    from braindecode.models import Labram
    torch.manual_seed(seed)
    if pretrained:
        m = Labram.from_pretrained("braindecode/labram-pretrained")
    else:
        m = Labram(n_times=SUB_SAMPLES, n_outputs=0, n_chans=len(CH_NAMES))
    return m.to(device).eval()


@torch.no_grad()
def embed(model, x, device="cuda", bs=32):
    """processed7 epochs [n,7,3000] -> [n, d] pooled over two 15 s sub-windows."""
    x = np.asarray(x[:, EEG_IDX], dtype=np.float32)
    r = resample_poly(x, DST_HZ // SRC_HZ, 1, axis=-1) / UV_SCALE      # [n,4,6000]
    n, c, t = r.shape
    k = t // SUB_SAMPLES
    sub = r.reshape(n, c, k, SUB_SAMPLES).transpose(0, 2, 1, 3).reshape(n * k, c, SUB_SAMPLES)
    out = []
    for i in range(0, len(sub), bs):
        o = model(torch.tensor(sub[i:i + bs]).to(device), ch_names=CH_NAMES)
        out.append(o.float().cpu().numpy())
    return np.concatenate(out).reshape(n, k, -1).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["labram", "labram_random"])
    ap.add_argument("--bs", type=int, default=32)
    # --source/--out let the same builder serve the external corpora, whose
    # arrays live outside data/processed7 (see EXTERNAL_VALIDATION_INSTRUCTIONS.md)
    ap.add_argument("--source", default=None,
                    help="directory of per-recording npz with an 'x' array; defaults to processed7")
    ap.add_argument("--out", default=None,
                    help="output directory; defaults to data/cbramod_emb/<variant>")
    a = ap.parse_args()

    src_dir = os.path.abspath(a.source) if a.source else P7
    out_dir = os.path.abspath(a.out) if a.out else os.path.join(REPO, "data", "cbramod_emb", a.variant)
    os.makedirs(out_dir, exist_ok=True)
    m = encoder(pretrained=(a.variant == "labram"))

    # external corpora do not use SN<k> names, so glob everything and sort plainly
    files = sorted(glob.glob(os.path.join(src_dir, "*.npz")))
    if not files:
        raise SystemExit("no .npz under %s" % src_dir)
    print("source: %s  (%d recordings)" % (src_dir, len(files)), flush=True)
    t0 = time.time(); done = 0
    for p in files:
        sid = os.path.basename(p)[:-4]
        dst = os.path.join(out_dir, sid + ".npz")
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            done += 1; continue
        d = np.load(p)
        if "x" not in d:
            print("[skip] %s: no raw 'x' array (external caches must carry it)" % sid, flush=True)
            continue
        e = embed(m, d["x"].astype(np.float32), bs=a.bs)
        assert len(e) == len(d["y"]), (e.shape, d["y"].shape)
        np.savez_compressed(dst, emb=e.astype(np.float16), y=d["y"])
        done += 1
        print("[ok] %-6s %5d epochs -> %s" % (sid, len(e), e.shape), flush=True)
    print("\n=== %s: %d subjects in %.1f min -> %s ==="
          % (a.variant, done, (time.time() - t0) / 60, os.path.relpath(out_dir, REPO)))


if __name__ == "__main__":
    main()
