"""Lesion-aware input block for MM-Net.

MM-Net itself is unchanged: same encoders, fusion, BiLSTM, bypass, heads and
training. This module only changes what the neural stream is given, in three
steps that can be switched on one at a time:

1. IPSILESIONAL ORDERING ("mirror"). The four EEG derivations are recorded as
   right/left pairs -- C4:M1 (c0) and O2:M1 (c2) over the right hemisphere,
   C3:M2 (c1) and O1:M2 (c3) over the left. For a patient whose infarct is in the
   left hemisphere the per-channel columns of the 188 engineered features are
   swapped c0<->c1 and c2<->c3, so for every lateralised patient c0/c2 are the
   lesioned hemisphere and c1/c3 the intact one. Right-sided, bilateral and
   unknown lesions keep the recorded order (right first), which is also the
   majority side, so nothing changes for them.

2. HEMISPHERIC ASYMMETRY ("asym"). Per epoch, for the central and the occipital
   pair, (ipsi - contra) / (ipsi + contra) of the relative delta, theta, alpha,
   sigma and beta power and of spindle density, plus the log-power ratio -- an
   epoch-level brain symmetry index. Computed from the RAW feature values, before
   the per-patient z-score that load_data applies, and left unscaled (the indices
   are bounded), so a patient's stable asymmetry is kept rather than centred away.

3. LESION CONTEXT ("context"). Four per-patient constants appended to every epoch:
   lateralised / bilateral / side unknown, and NIHSS / 10. Fixed scaling, no data
   statistics, so nothing crosses a fold boundary.

Lesion side comes from subject_description.xlsx: the `Side` column where filled,
otherwise the free-text Diagnosis / vascular-territory fields (rt/right, lt/left).
Row i of the workbook is SN(i+1), the same positional mapping clinical.py uses.
"""
import os
import re

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XLSX = os.path.join(REPO, "data", "full100", "subject_description.xlsx")
FE_DIR = os.path.join(REPO, "data", "mm_features")

STEPS = ("mirror", "asym", "context")
PAIRS = ((0, 1), (2, 3))                     # (right, left): central, occipital
RATIO_FEATS = ("delta_rel", "theta_rel", "alpha_rel", "sigma_rel", "beta_rel", "spindle_dens")


XLSX_URL = "https://ndownloader.figshare.com/files/55163861"   # Figshare 29253068, CC BY 4.0


def _ensure_xlsx():
    if not os.path.exists(XLSX):
        import urllib.request
        os.makedirs(os.path.dirname(XLSX), exist_ok=True)
        print("fetching subject_description.xlsx from Figshare", flush=True)
        urllib.request.urlretrieve(XLSX_URL, XLSX)


def lesion_table():
    """{sid: (side, nihss)} with side in 'R', 'L', 'B' or None."""
    _ensure_xlsx()
    df = pd.read_excel(XLSX)
    out = {}
    for i, r in df.iterrows():
        s = str(r["Side"]).strip().lower()
        side = {"r": "R", "right": "R", "l": "L", "left": "L",
                "b": "B", "bilateral": "B"}.get(s)
        if side is None:
            t = (str(r["Diagnosis_B.1"]) + " " + str(r["terri_vasc_B.1.a"])).lower()
            rr = bool(re.search(r"\b(rt|right)\b", t))
            ll = bool(re.search(r"\b(lt|left)\b", t))
            side = "B" if (rr and ll) else "R" if rr else "L" if ll else None
        nihss = r["NIHSS_scale"]
        out[i + 1] = (side, float(nihss) if pd.notna(nihss) else 0.0)
    return out


def _mirror_index(names):
    """Column permutation that swaps c0<->c1 and c2<->c3 in the 188 features."""
    pos = {n: i for i, n in enumerate(names)}
    swap = {"_c0": "_c1", "_c1": "_c0", "_c2": "_c3", "_c3": "_c2"}
    idx = []
    for n in names:
        tgt = n
        for a, b in swap.items():
            if n.endswith(a):
                tgt = n[: -len(a)] + b
                break
        idx.append(pos[tgt])
    return np.asarray(idx)


def _asym(raw, names, left):
    pos = {n: i for i, n in enumerate(names)}
    cols = []
    for r, l in PAIRS:
        ip, co = (l, r) if left else (r, l)
        for f in RATIO_FEATS:
            a = raw[:, pos["%s_c%d" % (f, ip)]]
            b = raw[:, pos["%s_c%d" % (f, co)]]
            cols.append(np.clip((a - b) / (np.abs(a) + np.abs(b) + 1e-6), -1, 1))
        lr = raw[:, pos["logtotal_c%d" % ip]] - raw[:, pos["logtotal_c%d" % co]]
        cols.append(np.clip(lr / 2.0, -2, 2))
    return np.stack(cols, 1).astype(np.float32)


def augment(data, names, steps):
    """data: {sid: (fe, fc, y, a)} whose fe starts with the 188 engineered columns.

    Returns (new data, number of columns added).
    """
    steps = set(steps)
    bad = steps - set(STEPS)
    if bad:
        raise ValueError("unknown lesion steps %s" % sorted(bad))
    table = lesion_table()
    perm = _mirror_index(names)
    out, added = {}, 0
    for sid, (fe, fc, y, a) in data.items():
        side, nihss = table.get(sid, (None, 0.0))
        left = side == "L"
        fe = fe.copy()
        if "mirror" in steps and left:
            fe[:, :188] = fe[:, perm]
        extra = []
        if "asym" in steps:
            raw = np.nan_to_num(np.load(os.path.join(FE_DIR, "SN%d.npz" % sid))["Feeg"]).astype(np.float32)
            if len(raw) != len(y):
                raise ValueError("SN%d: %d raw feature rows vs %d labels" % (sid, len(raw), len(y)))
            extra.append(_asym(raw, names, left))
        if "context" in steps:
            ctx = np.array([side in ("L", "R"), side == "B", side is None, nihss / 10.0], np.float32)
            extra.append(np.tile(ctx, (len(y), 1)))
        if extra:
            fe = np.concatenate([fe] + extra, axis=1)
        added = fe.shape[1] - data[sid][0].shape[1]
        out[sid] = (fe.astype(np.float32), fc, y, a)
    return out, added


def install(arms_mod, C, steps):
    """Wrap arms.load_arm so the lesion columns are added before widths are patched.

    Returns a function that undoes the wrap.
    """
    orig = arms_mod.load_arm

    def load_arm(C_, variant):
        data, dim = orig(C_, variant)
        if not variant.split("+")[0] == "A":
            raise ValueError("lesion block expects the 188 engineered features first")
        data, added = augment(data, C_.FEAT_NAMES, steps)
        return data, dim + added

    arms_mod.load_arm = load_arm
    return lambda: setattr(arms_mod, "load_arm", orig)
