"""Healthy-hemisphere emphasis for patients with a one-sided lesion.

Builds on lesion.py's ipsilesional ordering: after it, for every patient with a left
or right lesion, EEG channels c0 (central) and c2 (occipital) carry the LESIONED
hemisphere and c1 / c3 the INTACT one. Each EEG channel has the same 28 engineered
features in the same order, so the hemispheres can be exchanged column for column.

Modes (applied only to left- and right-sided patients; bilateral and unknown-side
patients keep their recorded features):
    healthy-copy   the lesioned hemisphere's 56 feature columns are replaced by the
                   intact hemisphere's, so the model reads only undamaged cortex
    damaged-drop   the lesioned hemisphere's 56 columns are set to 0, i.e. the
                   patient's own mean, since load_data z-scores features per patient

Only the 188 engineered features are changed. The LaBraM embedding mixes all four
EEG channels and is left as it is; EOG, EMG and the cardiorespiratory stream are
untouched. The network, training and folds are those of run_lesion_aware.py.
"""
import numpy as np

import lesion

MODES = ("healthy-copy", "damaged-drop")
PAIRS = ((0, 1), (2, 3))                  # (lesioned, intact) after ipsilesional ordering


def _cols(names, ch):
    return [i for i, n in enumerate(names) if n.endswith("_c%d" % ch)]


def _pairs(names):
    out = []
    for dmg, ok in PAIRS:
        a, b = _cols(names, dmg), _cols(names, ok)
        if [names[i][:-3] for i in a] != [names[i][:-3] for i in b]:
            raise ValueError("c%d and c%d feature lists do not line up" % (dmg, ok))
        out.append((a, b))
    return out


def install(arms_mod, C, mode):
    """Wrap arms.load_arm: ipsilesional ordering, then the healthy-hemisphere mode."""
    if mode not in MODES:
        raise ValueError("unknown mode %r" % mode)
    orig = arms_mod.load_arm
    table = lesion.lesion_table()
    pairs = _pairs(C.FEAT_NAMES)

    def load_arm(C_, variant):
        data, dim = orig(C_, variant)
        if not variant.split("+")[0] == "A":
            raise ValueError("lesion focus expects the 188 engineered features first")
        data, added = lesion.augment(data, C_.FEAT_NAMES, ("mirror",))
        out = {}
        for sid, (fe, fc, y, a) in data.items():
            if table.get(sid, (None, 0.0))[0] in ("L", "R"):
                fe = fe.copy()
                for dmg, ok in pairs:
                    fe[:, dmg] = fe[:, ok] if mode == "healthy-copy" else 0.0
            out[sid] = (fe.astype(np.float32), fc, y, a)
        return out, dim + added

    arms_mod.load_arm = load_arm
    return lambda: setattr(arms_mod, "load_arm", orig)
