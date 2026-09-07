"""Subject-level clinical covariates for the respiratory head.

Motivation. The paper's own learning curve shows staging has reached this
cohort's ceiling while respiratory detection has not -- it was still climbing at
76 training patients. So the head with headroom is the respiratory one, and the
metadata contains, at 94-98% completeness, exactly the clinical predictors of
obstructive sleep apnoea that a sleep physician would use: BMI, neck
circumference, abdominal circumference, Mallampati score, age and sex. Those are
the components of STOP-BANG.

AHI IS DELIBERATELY EXCLUDED. It is derived from the same respiratory events the
model is being asked to detect, so using it as an input would be circular and a
reviewer would say so immediately. Every covariate here is an independent
clinical measurement taken before any scoring.

Leakage. Covariates are centred and scaled by FIXED clinical reference values,
never by statistics computed from the data, so nothing crosses the fold boundary.
Missing values become 0 (the reference point) and are flagged by an explicit
indicator column, so the model can tell "average" from "unknown" instead of
silently treating them as the same.

Subject mapping. The workbook's Annonymized_Name column is inconsistent: 90 rows
are SN<k>.edf and 10 are AN1..AN10 sitting in positions 11-20. Row position i
maps to SN(i+1), verified exact for all 90 SN-labelled rows, which is the same
positional mapping the published supplementary notebook uses.
"""
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XLSX = os.path.join(REPO, "data", "full100", "subject_description.xlsx")

# name -> (column, centre, scale). Centres are clinical reference points, not
# data statistics, so no information crosses the train/test boundary.
COVARIATES = {
    "age":        ("Age_1.2",           55.0, 20.0),
    "sex":        ("sex",                1.5,  0.5),   # coded 1/2 -> -1/+1
    "bmi":        ("BMI2.g.1",          25.0, 10.0),
    "neck":       ("Neck_Cir2.g.2",     38.0,  6.0),
    "abd":        ("Abd_cir2.g.3",      95.0, 20.0),
    "mallampati": ("Mallampati_Score",   2.5,  1.5),
}
# excluded on purpose -- circular with the respiratory target
EXCLUDED = {"AHI_1_B": "derived from the events being detected"}

# Physiologically plausible ranges. The workbook contains data-entry errors that
# would otherwise dominate a scaled feature: BMI up to 106, neck circumference
# down to 1 cm, and Mallampati 0 on a scale that runs 1-4. Values outside these
# bounds are treated as MISSING rather than clipped, because a neck of 1 cm is
# not a small neck -- it is an unknown one, and the missingness indicator says so
# honestly.
PLAUSIBLE = {
    "age":        (18.0, 100.0),
    "sex":        (1.0, 2.0),
    "bmi":        (12.0, 60.0),
    "neck":       (25.0, 60.0),
    "abd":        (50.0, 160.0),
    "mallampati": (1.0, 4.0),
}


def load_table(names=None):
    """-> {subject_id: np.array([...covariates..., ...missing flags...], float32)}"""
    names = list(COVARIATES) if names is None else list(names)
    df = pd.read_excel(XLSX)
    out = {}
    for i in range(len(df)):
        sid = i + 1                      # positional mapping, verified exact
        vals, miss = [], []
        for n in names:
            col, centre, scale = COVARIATES[n]
            v = pd.to_numeric(pd.Series([df.iloc[i][col]]), errors="coerce").iloc[0]
            lo, hi = PLAUSIBLE[n]
            if pd.isna(v) or not (lo <= float(v) <= hi):
                vals.append(0.0); miss.append(1.0)
            else:
                vals.append(float((v - centre) / scale)); miss.append(0.0)
        out[sid] = np.array(vals + miss, dtype=np.float32)
    return out, names


def coverage(names=None):
    """Per-covariate completeness, for reporting rather than modelling."""
    names = list(COVARIATES) if names is None else list(names)
    df = pd.read_excel(XLSX)
    rep = {}
    for n in names:
        col = COVARIATES[n][0]
        v = pd.to_numeric(df[col], errors="coerce")
        lo, hi = PLAUSIBLE[n]
        ok = v.between(lo, hi)
        rep[n] = dict(n=int(ok.sum()), missing=int((~ok).sum()),
                      implausible=int((v.notna() & ~ok).sum()),
                      lo=float(v[ok].min()), hi=float(v[ok].max()))
    return rep
