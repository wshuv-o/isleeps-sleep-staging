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


def _load_one(C, name):
    """One representation -> ({sid: [n, d] float32}, d)."""
    if name == "A":
        return {sid: C.DATA[sid][0] for sid in C.SUBS}, 188
    out, dim = {}, None
    for f in sorted(glob.glob(os.path.join(EMB, name, "SN*.npz")),
                    key=lambda p: int(os.path.basename(p)[2:-4])):
        sid = int(os.path.basename(f)[2:-4])
        if sid in C.DUP or sid not in C.DATA:
            continue
        d = np.load(f)
        e = d["emb"].astype(np.float32).reshape(len(d["emb"]), -1)
        y = C.DATA[sid][2]
        if len(e) != len(y):
            raise ValueError("SN%d/%s: %d embedded epochs vs %d in mm_features"
                             % (sid, name, len(e), len(y)))
        if not np.array_equal(d["y"], y):
            raise ValueError("SN%d/%s: label mismatch against mm_features" % (sid, name))
        out[sid] = _zscore(e)
        dim = e.shape[1]
    if not out:
        raise ValueError("no subjects for representation %r under %s" % (name, EMB))
    return out, dim


def load_arm(C, variant):
    """variant: '+'-joined representation names.

    'A'                     the published 188 engineered features
    'pretrained'            frozen CBraMod embedding
    'labram'                frozen LaBraM embedding
    'A+pretrained'          their concatenation, and so on for any combination

    Each representation is z-scored per subject exactly as load_data() z-scores
    the 188 features, so no representation is advantaged by scale before they are
    concatenated. Labels always come from mm_features, so only the EEG
    representation differs between arms.

    The combination exists because arm B matched arm A on accuracy while beating
    it on macro-F1: equal accuracy with a different error profile implies the two
    carry different information. If a union beats its constituents they are
    complementary; if it matches them, they encode the same thing.
    """
    parts = [p for p in variant.split("+") if p]
    if len(parts) == 1 and parts[0] == "A":
        return {sid: C.DATA[sid] for sid in C.SUBS}, 188
    loaded, dims = [], []
    for name in parts:
        rep, d = _load_one(C, name)
        loaded.append(rep); dims.append(d)
    common = sorted(set.intersection(*(set(r) for r in loaded)) & set(C.SUBS))
    data = {}
    for sid in common:
        fe = np.concatenate([r[sid] for r in loaded], axis=1).astype(np.float32)
        _, fc, y, a = C.DATA[sid]
        if len(fe) != len(y):
            raise ValueError("SN%d: %d rows after concatenation vs %d labels" % (sid, len(fe), len(y)))
        data[sid] = (fe, fc, y, a)
    return data, int(sum(dims))


# mmnet_core hardcodes the 188-wide EEG block in four places -- windows(),
# subj_infer(), infer_arrays() and subj_embed() -- each as a zero-pad and a
# reshape. Re-typing those functions here would risk diverging from the code
# that produced the published result, so instead their own source is recompiled
# with the width literal substituted. The logic is therefore identical by
# construction, and the substitution count is asserted so a future edit to
# mmnet_core cannot silently change what gets patched.
_WIDTH_PATCHED = {"windows": 1, "subj_infer": 2, "infer_arrays": 2, "subj_embed": 2}
# the cardio width literal (14) sits in exactly the same places as the EEG one
_CARD_PATCHED = {"windows": 1, "subj_infer": 2, "infer_arrays": 2, "subj_embed": 2}


def _recompile_with_width(C, n_eeg, n_card=None):
    """Substitute the hardcoded feature widths in mmnet_core's own source.

    n_card is only patched when the cardio stream has been widened by clinical
    covariates; leaving it None keeps the published 14.
    """
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
        if n_card is not None:
            exp_c = _CARD_PATCHED[name]
            src, nc = re.subn(r"(?<![\w.])14(?![\w.])", str(n_card), src)
            if nc != exp_c:
                raise RuntimeError("%s: replaced %d occurrences of 14, expected %d -- "
                                   "mmnet_core changed, re-check the patch" % (name, nc, exp_c))
        ns = dict(C.__dict__)
        exec(compile(src, "<%s:eeg=%d,card=%s>" % (name, n_eeg, n_card), "exec"), ns)
        out[name] = ns[name]
    return out


class arm:
    """Context manager: run mmnet_core with one arm's EEG representation.

        with arm(C, "pretrained"):
            r = C.run_10fold(fusion="concat", seed=42)

    Restores the published configuration on exit, so arms cannot leak into each
    other and Arm A always means the same thing.
    """

    def __init__(self, C, variant, covars=None):
        """covars: None, or a list of clinical covariate names from clinical.py.

        Covariates are appended to the CARDIO stream, not the EEG one, because
        they are predictors of sleep-disordered breathing and the respiratory
        head is the one the learning curve says still has headroom. They reach
        that head both through card_enc and through the validated bypass.
        """
        self.C, self.variant, self.covars = C, variant, covars

    def __enter__(self):
        C = self.C
        self._saved = {"DATA": C.DATA, "MMFeatureNet": C.MMFeatureNet}
        for name in _WIDTH_PATCHED:
            self._saved[name] = getattr(C, name)
        data, dim = load_arm(C, self.variant)
        n_card = None
        if self.covars is not None:
            import clinical
            table, used = clinical.load_table(self.covars)
            n_card = 14 + 2 * len(used)          # values + missingness indicators
            wide = {}
            for sid, (fe, fc, y, a) in data.items():
                if sid not in table:
                    raise KeyError("SN%d has no metadata row" % sid)
                cov = np.tile(table[sid], (len(y), 1))
                wide[sid] = (fe, np.concatenate([fc, cov], axis=1).astype(np.float32), y, a)
            data = wide
            self.covar_names = used
        if self.variant != "A" or n_card is not None:
            C.DATA = data
            for name, fn in _recompile_with_width(C, dim, n_card).items():
                setattr(C, name, fn)
            net = self._saved["MMFeatureNet"]
            kw_fixed = {"n_eeg": dim}
            if n_card is not None:
                kw_fixed["n_card"] = n_card
            C.MMFeatureNet = lambda **kw: net(**{**kw_fixed, **kw})
        self.dim, self.n_card = dim, (n_card if n_card is not None else 14)
        return self

    def __exit__(self, *exc):
        for name, obj in self._saved.items():
            setattr(self.C, name, obj)
        return False
