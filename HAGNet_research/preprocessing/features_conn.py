"""
features_conn.py -- inter-hemispheric CONNECTIVITY features: the one EEG signal our
per-channel features miss, and the physics of both the Brain Symmetry Index and our
spindle-asymmetry biomarker. Stroke disrupts how the hemispheres communicate, so
coherence / phase-locking between homologous derivations is a physiologically motivated,
disease-relevant addition (not a trendy bolt-on).

For the 7-channel montage [C4:M1(0) C3:M2(1) O2:M1(2) O1:M2(3) E1:M2(4) E2:M2(5) EMG(6)]:
  homologous pairs   : (C4,C3), (O2,O1)          -> inter-hemispheric
  within-hemisphere  : (C4,O2) right, (C3,O1) left
Per pair: band-limited magnitude-squared coherence (delta/theta/alpha/sigma/beta) +
sigma-band phase-locking value (PLV, the spindle band). ~ (4 pairs x 5 bands) + 2 PLV.
"""
import numpy as np
from scipy.signal import coherence, butter, sosfiltfilt, hilbert

BANDS = [("delta", 0.5, 4), ("theta", 4, 8), ("alpha", 8, 12), ("sigma", 12, 16), ("beta", 16, 30)]
PAIRS = [("C4C3", 0, 1), ("O2O1", 2, 3), ("C4O2", 0, 2), ("C3O1", 1, 3)]   # inter-hemi + within-hemi


def _plv(a, b, lo, hi, fs):
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    pa = np.angle(hilbert(sosfiltfilt(sos, a, axis=-1), axis=-1))
    pb = np.angle(hilbert(sosfiltfilt(sos, b, axis=-1), axis=-1))
    return np.abs(np.exp(1j * (pa - pb)).mean(-1))          # [n]


def connectivity_features(x, fs=100):
    """x [n,7,3000] -> connectivity features [n, F], names."""
    n = x.shape[0]; feats, names = [], []
    for pname, i, j in PAIRS:
        f, Cxy = coherence(x[:, i, :], x[:, j, :], fs=fs, nperseg=256, axis=-1)   # [n, nf]
        for bname, lo, hi in BANDS:
            m = (f >= lo) & (f < hi)
            feats.append(Cxy[:, m].mean(-1)); names.append(f"coh_{pname}_{bname}")
    for pname, i, j in [("C4C3", 0, 1), ("O2O1", 2, 3)]:    # spindle-band PLV, homologous only
        feats.append(_plv(x[:, i, :], x[:, j, :], 12, 16, fs)); names.append(f"plv_sigma_{pname}")
    F = np.nan_to_num(np.stack(feats, axis=1).astype(np.float32))
    return F, names


def extract_features_v3(x, fs=100):
    """v2 (spectral + event) + inter-hemispheric connectivity. x [n,7,3000] -> [n,F], names."""
    from features_v2 import extract_features_v2
    Fb, nb = extract_features_v2(x, fs=fs)
    Fc, nc = connectivity_features(x, fs=fs)
    return np.concatenate([Fb, Fc], axis=1), nb + nc


if __name__ == "__main__":
    import os
    d = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "processed7", "SN1.npz"), allow_pickle=True)
    F, names = extract_features_v3(d["x"].astype(np.float32))
    conn = [nm for nm in names if nm.startswith(("coh_", "plv_"))]
    print("x", d["x"].shape, "-> v3 features", F.shape, f"(+{len(conn)} connectivity)")
    print("connectivity:", conn)
