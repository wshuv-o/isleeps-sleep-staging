"""Precompute frozen CBraMod embeddings for every processed7 epoch.

Arms B and D freeze the encoder, so its output can be computed once instead of
re-run every training epoch. Two caches are built from identical code paths --
'pretrained' (the released checkpoint) and 'random' (same architecture, random
init) -- so Arm D's control differs from Arm B in exactly one thing.

Input  : data/processed7/SN<k>.npz   x [n,7,3000] uV @100 Hz
Output : data/cbramod_emb/<variant>/SN<k>.npz
           emb [n,7,200] float16   patch-pooled encoder output
           y   [n]                 sleep stage, copied through for alignment
Adaptation applied: resample_poly 100->200 Hz, 30x200 one-second patches,
uV/100 scaling (CBraMod's own loaders use seq/100).
"""
import argparse, glob, os, sys, time
import numpy as np, torch
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.cbramod import CBraMod

REPO = r"d:\proc\isleeps-sleep-staging"
HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"
SCALE = 100.0          # CBraMod loaders return seq/100; processed7 is raw uV, never z-scored


def encoder(variant, seed=42):
    torch.manual_seed(seed)
    m = CBraMod().to(DEV)
    if variant == "pretrained":
        sd = torch.load(os.path.join(HERE, "cbramod.pth"), map_location=DEV, weights_only=True)
        missing, unexpected = m.load_state_dict(sd, strict=True), None
    elif variant != "random":
        raise SystemExit("variant must be pretrained|random")
    m.proj_out = torch.nn.Identity()
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["pretrained", "random"])
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--pool", default="mean", choices=["mean", "mean_std", "mean_std_minmax"],
                    help="how the 30 one-second patches are collapsed per epoch")
    ap.add_argument("--suffix", default="", help="cache subdirectory suffix")
    a = ap.parse_args()

    out_dir = os.path.join(REPO, "data", "cbramod_emb", a.variant + a.suffix)
    os.makedirs(out_dir, exist_ok=True)
    m = encoder(a.variant)

    files = sorted(glob.glob(os.path.join(REPO, "data", "processed7", "SN*.npz")),
                   key=lambda p: int(os.path.basename(p)[2:-4]))
    t0 = time.time(); done = 0
    for p in files:
        sid = os.path.basename(p)[:-4]
        dst = os.path.join(out_dir, sid + ".npz")
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            done += 1; continue
        d = np.load(p)
        x = d["x"].astype(np.float32); y = d["y"]; n = len(x)
        xr = resample_poly(x, 2, 1, axis=-1).reshape(n, 7, 30, 200) / SCALE
        chunks = []
        with torch.no_grad():
            for i in range(0, n, a.bs):
                o = m(torch.tensor(xr[i:i + a.bs]).to(DEV))     # [b,7,30,200]
                # Mean over the 30 one-second patches discards all within-epoch
                # temporal structure: a K-complex in second 3 and one in second 27
                # produce the same vector. Higher-order statistics keep some of it
                # -- std captures how much the second-by-second representation
                # varies across the epoch, min/max capture transients that an
                # average washes out.
                parts = [o.mean(dim=2)]
                if a.pool in ("mean_std", "mean_std_minmax"):
                    parts.append(o.std(dim=2))
                if a.pool == "mean_std_minmax":
                    parts.append(o.amax(dim=2)); parts.append(o.amin(dim=2))
                chunks.append(torch.cat(parts, dim=-1).half().cpu().numpy())
        emb = np.concatenate(chunks)
        mult = {"mean": 1, "mean_std": 2, "mean_std_minmax": 4}[a.pool]
        assert emb.shape == (n, 7, 200 * mult) and len(y) == n, (emb.shape, len(y))
        np.savez_compressed(dst, emb=emb, y=y)
        done += 1
        print("[ok] %-6s %5d epochs -> %s" % (sid, n, emb.shape), flush=True)
    print("\n=== %s: %d subjects in %.1f min -> %s ==="
          % (a.variant, done, (time.time() - t0) / 60, os.path.relpath(out_dir, REPO)))


if __name__ == "__main__":
    main()
