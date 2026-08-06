"""
features_new.py -- NEW information, not a re-slice of what we already have.

Every lever that failed so far (decoding, calibration, thresholds, context width)
re-used the same 188 spectral/Hjorth/event features. The per-class-bias experiment
showed N1 gains are paid for one-for-one elsewhere, i.e. the features genuinely do
not separate N1. So this block adds *new* signal, drawn from Sanei & Chambers,
"EEG Signal Processing":

  §2.7  nonlinear / complexity : permutation entropy, Higuchi FD, Petrosian FD,
                                 Lempel-Ziv complexity
  §2.5.1 wavelet transform     : 5-level band energies + wavelet entropy
                                 (transient-sensitive where Welch PSD smears)
  §6.5   sleep graphoelements  : K-complex / vertex-wave transient counts
  N1-specific                  : slow (SEM, 0.3-1 Hz) vs rapid (1-5 Hz) eye movement,
                                 the exact N1-vs-REM confusion in our matrix

Deliberately omitted: approximate/sample entropy (O(N^2), too slow for 95k epochs
and largely redundant with permutation entropy).
"""
import math
import numpy as np
from scipy.signal import butter, filtfilt

FS = 100

try:
    import pywt
    HAVE_PYWT = True
except Exception:
    HAVE_PYWT = False


# ----------------------------------------------------------------- complexity
def perm_entropy(x, order=3, delay=1):
    """normalised permutation entropy (Bandt-Pompe)"""
    n = len(x) - (order - 1) * delay
    if n <= 1:
        return 0.0
    idx = np.arange(order) * delay
    emb = x[np.arange(n)[:, None] + idx[None, :]]
    perms = np.argsort(emb, axis=1)
    # encode each permutation pattern as an integer
    mult = order ** np.arange(order)
    codes = (perms * mult).sum(1)
    _, cnt = np.unique(codes, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log(p)).sum() / np.log(math.factorial(order)))


def higuchi_fd(x, kmax=6, decim=3):
    """Higuchi fractal dimension (decimated: 3000 -> 1000 samples, ~3x faster,
    FD is scale-robust so the estimate is unaffected in practice)"""
    x = x[::decim]
    N = len(x); L = []
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            Lm = np.abs(np.diff(x[idx])).sum() * (N - 1) / (len(idx) - 1) / k
            Lk.append(Lm)
        if Lk:
            L.append(np.log(np.mean(Lk) + 1e-12))
    if len(L) < 2:
        return 0.0
    k = np.log(1.0 / np.arange(1, len(L) + 1))
    return float(np.polyfit(k, L, 1)[0])


def petrosian_fd(x):
    d = np.diff(x)
    nd = int((d[1:] * d[:-1] < 0).sum())          # sign changes in derivative
    n = len(x)
    return float(np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * nd + 1e-12))))


def lziv(x):
    """Lempel-Ziv complexity of the median-binarised signal"""
    s = (x > np.median(x)).astype(np.uint8)
    n = len(s); i, k, l, c, kmax = 0, 1, 1, 1, 1
    while True:
        if i + k > n or l + k > n:
            break
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1; break
        else:
            kmax = max(kmax, k); i += 1
            if i == l:
                c += 1; l += kmax; i = 0; kmax = 1
            k = 1
    return float(c * np.log2(n + 1e-12) / (n + 1e-12))


# ----------------------------------------------------------------- wavelet
def wavelet_feats(x):
    """5-level band energies (log) + wavelet entropy"""
    if HAVE_PYWT:
        cs = pywt.wavedec(x, "db4", level=5)
    else:                                          # filter-bank fallback
        bands = [(0.5, 2), (2, 4), (4, 8), (8, 16), (16, 30), (30, 45)]
        cs = []
        for lo, hi in bands:
            b, a = butter(4, [lo / (FS / 2), min(hi, 49) / (FS / 2)], btype="band")
            cs.append(filtfilt(b, a, x))
    e = np.array([float(np.sum(c ** 2)) for c in cs]) + 1e-12
    p = e / e.sum()
    went = float(-(p * np.log(p)).sum())
    return list(np.log(e)) + [went]


# ----------------------------------------------------------------- transients
def kcomplex_count(x):
    """large biphasic slow deflections (K-complex / vertex sharp wave proxy)"""
    b, a = butter(4, [0.5 / (FS / 2), 4 / (FS / 2)], btype="band")
    d = filtfilt(b, a, x)
    thr = 2.5 * np.std(d) + 1e-9
    neg = d < -thr
    starts = np.flatnonzero(np.diff(neg.astype(int)) == 1)
    return float(len(starts)), float(np.abs(d).max())


def eog_slow_rapid(x):
    """N1 keys on SLOW rolling eye movement; REM keys on rapid. Separate them."""
    def bp(lo, hi):
        b, a = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band")
        return filtfilt(b, a, x)
    slow = bp(0.3, 1.0); rapid = bp(1.0, 5.0)
    es, er = float(np.mean(slow ** 2)), float(np.mean(rapid ** 2))
    return [np.log(es + 1e-12), np.log(er + 1e-12), np.log((es + 1e-12) / (er + 1e-12))]


# ----------------------------------------------------------------- per-epoch
EEG_CH, EOG_CH, EMG_CH = [0, 1, 2, 3], [4, 5], 6


def epoch_features(ep):
    """ep: [7, 3000] -> 1-D feature vector of NEW features only"""
    # NOTE: Lempel-Ziv deliberately excluded. The naive LZ76 is O(n^2) in Python
    # (~2.4 s per channel, i.e. ~13 days for this dataset) and permutation entropy
    # already captures the complexity axis at ~2 ms.
    f = []
    for c in EEG_CH:
        x = ep[c]
        f += [perm_entropy(x), higuchi_fd(x), petrosian_fd(x)]
        f += wavelet_feats(x)
        kc, kamp = kcomplex_count(x)
        f += [kc, np.log(kamp + 1e-12)]
    for c in EOG_CH:
        x = ep[c]
        f += eog_slow_rapid(x)
        f += [perm_entropy(x), higuchi_fd(x)]
    x = ep[EMG_CH]
    f += [perm_entropy(x), higuchi_fd(x), petrosian_fd(x)]
    return np.asarray(f, dtype=np.float32)


def subject_features(X):
    """X: [n_epochs, 7, 3000] -> [n_epochs, D_new]"""
    return np.stack([epoch_features(X[i]) for i in range(len(X))]).astype(np.float32)


def n_features():
    return len(epoch_features(np.random.randn(7, 3000).astype(np.float32)))


if __name__ == "__main__":
    d = n_features()
    print(f"new feature dim = {d} per epoch  (pywt available: {HAVE_PYWT})")
