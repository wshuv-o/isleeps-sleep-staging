"""Swap mmnet_core's EEG feature path for a foundation-model encoder.

Section 4 of the handoff specifies replacing the eeg_enc MLP and leaving
everything else identical -- cardiorespiratory stream, fusion, BiLSTM, the
validated bypass, both heads -- so the encoder is the only variable. This module
does exactly that by rebinding mmnet_core's globals, without editing the file
that produced the published result.

Three things have to change together, and missing any one of them silently
breaks the comparison:

1. DATA. Feeg (188 engineered) is replaced by the CBraMod embedding, and given
   the SAME per-subject z-scoring load_data() applies to the 188 features. Without
   it Arm B is handicapped by feature scale rather than by information content.
   This is a separate normalisation from the uV/100 applied at the encoder input.
2. windows(). The original hardcodes np.zeros((k,188)) when padding a short
   window, so any other feature width raises on the last window of a subject
   whose epoch count is not a multiple of L.
3. MMFeatureNet. train_fold constructs it with the default n_eeg=188; the
   embedding is wider.

y and apnea always come from mm_features, so the labels are identical across
arms and only the EEG representation differs.
"""
import glob
import os

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMB = os.path.join(REPO, "data", "cbramod_emb")


def _zscore(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def load_arm(C, variant):
    """variant: 'A' (published 188 features), a cbramod_emb subdirectory name, or
    'A+<name>' to concatenate the engineered features with that embedding.

    The concatenation exists because arm B matched arm A on accuracy while beating
    it on macro-F1: same accuracy with a different error profile suggests the two
    representations carry different information. If the union beats both, they are
    complementary; if it matches them, they encode the same thing and the ceiling
    is a property of the signal rather than of either representation.
    """
    if variant == "A":
        return {s: C.DATA[s] for s in C.SUBS}, 188
    if variant.startswith("A+"):
        emb, dim = load_arm(C, variant[2:])
        data = {}
        for sid, (e, fc, y, a) in emb.items():
            feats = C.DATA[sid][0]          # already per-subject z-scored by load_data
            if len(feats) != len(e):
                raise ValueError("SN%d: %d feature rows vs %d embedding rows"
                                 % (sid, len(feats), len(e)))
            data[sid] = (np.concatenate([feats, e], axis=1), fc, y, a)
        return data, dim + 188
    data, dim = {}, None
    for f in sorted(glob.glob(os.path.join(EMB, variant, "SN*.npz")),
                    key=lambda p: int(os.path.basename(p)[2:-4])):
        sid = int(os.path.basename(f)[2:-4])
        if sid in C.DUP or sid not in C.DATA:
            continue
        d = np.load(f)
        e = d["emb"].astype(np.float32).reshape(len(d["emb"]), -1)
        _, fc, y, a = C.DATA[sid]
        # the embedding and the feature cache must describe the same epochs
        if len(e) != len(y):
            raise ValueError("SN%d: %d embedded epochs vs %d in mm_features" % (sid, len(e), len(y)))
        if not np.array_equal(d["y"], y):
            raise ValueError("SN%d: label mismatch between cbramod_emb and mm_features" % sid)
        data[sid] = (_zscore(e), fc, y, a)
        dim = e.shape[1]
    if not data:
        raise ValueError("no subjects found for variant %r under %s" % (variant, EMB))
    return data, dim


# mmnet_core hardcodes the 188-wide EEG block in four places -- windows(),
# subj_infer(), infer_arrays() and subj_embed() -- each as a zero-pad and a
# reshape. Re-typing those functions here would risk diverging from the code
# that produced the published result, so instead their own source is recompiled
# with the width literal substituted. The logic is therefore identical by
# construction, and the substitution count is asserted so a future edit to
# mmnet_core cannot silently change what gets patched.
_WIDTH_PATCHED = {"windows": 1, "subj_infer": 2, "infer_arrays": 2, "subj_embed": 2}


def _recompile_with_width(C, n_eeg):
    import inspect
    import re
    import textwrap
    out = {}
    for name, expected in _WIDTH_PATCHED.items():
        src = textwrap.dedent(inspect.getsource(getattr(C, name)))
        src, n = re.subn(r"(?<![\w.])188(?![\w.])", str(n_eeg), src)
        if n != expected:
            raise RuntimeError("%s: replaced %d occurrences of 188, expected %d -- "
                               "mmnet_core changed, re-check the patch" % (name, n, expected))
        ns = dict(C.__dict__)
        exec(compile(src, "<%s:width=%d>" % (name, n_eeg), "exec"), ns)
        out[name] = ns[name]
    return out


class arm:
    """Context manager: run mmnet_core with one arm's EEG representation.

        with arm(C, "pretrained"):
            r = C.run_10fold(fusion="concat", seed=42)

    Restores the published configuration on exit, so arms cannot leak into each
    other and Arm A always means the same thing.
    """

    def __init__(self, C, variant):
        self.C, self.variant = C, variant

    def __enter__(self):
        C = self.C
        self._saved = {"DATA": C.DATA, "MMFeatureNet": C.MMFeatureNet}
        for name in _WIDTH_PATCHED:
            self._saved[name] = getattr(C, name)
        data, dim = load_arm(C, self.variant)
        if self.variant != "A":
            C.DATA = data
            for name, fn in _recompile_with_width(C, dim).items():
                setattr(C, name, fn)
            net = self._saved["MMFeatureNet"]
            C.MMFeatureNet = lambda **kw: net(n_eeg=dim, **kw)
        self.dim = dim
        return self

    def __exit__(self, *exc):
        for name, obj in self._saved.items():
            setattr(self.C, name, obj)
        return False
