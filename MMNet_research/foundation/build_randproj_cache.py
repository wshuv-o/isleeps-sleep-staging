"""Random-projection baselines for the cardiorespiratory branch.

A randomly initialised 341 M-parameter MOMENT reached 0.7947 respiratory AUC
against 0.6977 for the 14 engineered features, so 88% of that branch's gain is
not pretraining. These two baselines pin down what it IS, and they need no
pretrained model at all.

  randproj_raw   raw cardio signal (7 x 750) -> fixed random matrix -> 1024-d
                 If this matches the random transformer, the depth and the 341 M
                 parameters are irrelevant and the effect is plain random
                 features over the raw signal.

  randproj_feat  the 14 engineered features -> fixed random matrix -> 1024-d
                 This is the control for the control. A random projection cannot
                 add information, so if widening 14 dimensions to 1024 also
                 helped, the gain would be an artefact of input width rather than
                 anything about the signal. It should land on the baseline, and
                 if it does not, every conclusion here needs revisiting.

Together they separate three explanations that the MOMENT runs alone cannot:
information in the raw signal, capacity of the projection, and pretraining.

Projections are drawn once with a fixed seed and shared across all subjects, so
this is a single deterministic feature map, not a per-subject fit -- nothing
crosses the train/test boundary.
"""
import argparse
import glob
import os
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MM = os.path.join(REPO, "data", "multimodal")
FE = os.path.join(REPO, "data", "mm_features")
OUT_DIM = 1024
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["randproj_raw", "randproj_feat"])
    ap.add_argument("--dim", type=int, default=OUT_DIM)
    a = ap.parse_args()

    out_dir = os.path.join(REPO, "data", "cardio_emb", a.variant)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(MM, "SN*.npz")),
                   key=lambda p: int(os.path.basename(p)[2:-4]))
    rng = np.random.RandomState(SEED)
    W = None
    t0 = time.time(); done = 0

    for p in files:
        sid = os.path.basename(p)[:-4]
        dst = os.path.join(out_dir, sid + ".npz")
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            done += 1; continue
        d = np.load(p)
        if a.variant == "randproj_raw":
            card = d["card"].astype(np.float32)                    # [n, 7, 750]
            x = card.reshape(len(card), -1)                        # [n, 5250]
        else:
            f = np.load(os.path.join(FE, sid + ".npz"))
            x = np.nan_to_num(f["Fcard"]).astype(np.float32)       # [n, 14]

        # Per-subject standardisation BEFORE projection, matching how the raw
        # cardio channels differ wildly in units (SpO2 in %, ECG in uV).
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)

        if W is None:
            # one fixed projection shared by every subject; scaled 1/sqrt(d_in)
            # so the output variance does not depend on the input width
            W = rng.normal(0.0, 1.0 / np.sqrt(x.shape[1]),
                           size=(x.shape[1], a.dim)).astype(np.float32)
            print("projection: %d -> %d, fixed seed %d" % (x.shape[1], a.dim, SEED), flush=True)

        e = np.tanh(x @ W)          # tanh keeps it a bounded nonlinear random feature map
        assert np.isfinite(e).all(), sid
        np.savez_compressed(dst, emb=e.astype(np.float16), y=d["y"])
        done += 1
        if done % 20 == 0:
            print("  %s  %d subjects" % (sid, done), flush=True)
    print("\n=== %s: %d subjects in %.1f min -> %s ==="
          % (a.variant, done, (time.time() - t0) / 60,
             os.path.relpath(out_dir, REPO)))


if __name__ == "__main__":
    main()
