"""
lesion_stratify.py — RQ1 / RQ3 preliminary analysis (the new, fundamental contribution).

Tests whether sleep microstructure is disrupted IPSILESIONALLY using a within-subject
contrast (each patient is their own control), and whether the asymmetry scales with NIHSS.

Microstructure proxies (per channel, relative power, robust to scale):
  - SPINDLE proxy : sigma-band (11-16 Hz) relative power over N2 epochs
  - SLOW-WAVE proxy: delta-band (0.5-4 Hz) relative power over N3 epochs

Channel hemispheres: C4:M1, O2:M1 = RIGHT ; C3:M2, O1:M2 = LEFT.
Ipsilesional = same hemisphere as the lesion (from subject_description.xlsx `Side`).

  KMP_DUPLICATE_LIB_OK=TRUE d:/EEG-TransNet/testenv/python.exe analysis/lesion_stratify.py
"""
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import wilcoxon, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "models"))
from datasets import load_subject, CHANNELS  # noqa

PROC = os.path.join(ROOT, "data", "processed")
META = os.path.join(ROOT, "data", "full100", "subject_description.xlsx")
RES = os.path.join(ROOT, "results"); os.makedirs(RES, exist_ok=True)
FIGDIR = os.path.join(ROOT, "presentation", "figures"); os.makedirs(FIGDIR, exist_ok=True)

RIGHT_CH = [CHANNELS.index("C4:M1"), CHANNELS.index("O2:M1")]
LEFT_CH = [CHANNELS.index("C3:M2"), CHANNELS.index("O1:M2")]
FS = 100


def lesion_sides():
    df = pd.read_excel(META)
    df["sid"] = df["Annonymized_Name"].astype(str).str.extract(r"SN(\d+)").astype(float)
    out = {}
    for _, r in df.iterrows():
        if pd.isna(r["sid"]):
            continue
        s = str(r.get("Side", "")).strip().lower()
        side = {"r": "R", "right": "R", "l": "L", "left": "L"}.get(s)
        if s in ("b", "bilateral"):
            side = "B"
        out[int(r["sid"])] = {"side": side, "nihss": r.get("NIHSS_scale", np.nan),
                              "mrs": r.get("MRS_scale", np.nan), "ahi": r.get("AHI_1_B", np.nan)}
    return out


def band_rel_power(x, stage_idx, lo, hi):
    """x [n,3000] for one channel, stage_idx: epoch indices to use. Returns mean relative power."""
    if len(stage_idx) == 0:
        return np.nan
    xx = x[stage_idx]
    f, p = welch(xx, fs=FS, nperseg=256, noverlap=128, axis=-1)  # [m, F]
    tot = p[:, (f >= 0.5) & (f < 30)].sum(-1) + 1e-12
    band = p[:, (f >= lo) & (f < hi)].sum(-1)
    return float(np.mean(band / tot))


def subject_microstructure(sid):
    x, y = load_subject(sid, normalize=False)        # x [n,4,3000], uV
    n2 = np.where(y == 2)[0]; n3 = np.where(y == 3)[0]
    sigma = [band_rel_power(x[:, c, :], n2, 11, 16) for c in range(4)]   # spindle proxy (N2)
    swa = [band_rel_power(x[:, c, :], n3, 0.5, 4) for c in range(4)]     # slow-wave proxy (N3)
    return np.array(sigma), np.array(swa), len(n2), len(n3)


def main():
    meta = lesion_sides()
    avail = sorted(int(os.path.basename(p)[2:-4]) for p in glob.glob(os.path.join(PROC, "SN*.npz")))
    rows = []
    for sid in avail:
        m = meta.get(sid)
        if not m or m["side"] not in ("L", "R"):
            continue
        sigma, swa, n_n2, n_n3 = subject_microstructure(sid)
        if n_n2 < 20:                    # spindle analysis needs enough N2 only
            continue
        ipsi = RIGHT_CH if m["side"] == "R" else LEFT_CH
        contra = LEFT_CH if m["side"] == "R" else RIGHT_CH
        rows.append({
            "sid": sid, "side": m["side"], "nihss": m["nihss"], "ahi": m["ahi"],
            "sigma_ipsi": np.nanmean(sigma[ipsi]), "sigma_contra": np.nanmean(sigma[contra]),
            # slow-wave only valid with enough N3
            "swa_ipsi": np.nanmean(swa[ipsi]) if n_n3 >= 5 else np.nan,
            "swa_contra": np.nanmean(swa[contra]) if n_n3 >= 5 else np.nan,
        })
    df = pd.DataFrame(rows).dropna(subset=["sigma_ipsi", "sigma_contra"])
    n = len(df)
    n_sw = df.dropna(subset=["swa_ipsi", "swa_contra"]).shape[0]
    print(f"unilateral-lesion subjects: spindle/N2 analysis N={n} "
          f"(R={(df.side=='R').sum()}, L={(df.side=='L').sum()}) | slow-wave/N3 analysis N={n_sw}")

    def report(name, ipsi_col, contra_col):
        sub = df.dropna(subset=[ipsi_col, contra_col])
        ipsi = sub[ipsi_col].values; contra = sub[contra_col].values
        d = ipsi - contra
        ai = (ipsi - contra) / (ipsi + contra)        # negative => ipsilesional reduction
        try:
            stat, pval = wilcoxon(ipsi, contra)
        except ValueError:
            stat, pval = np.nan, np.nan
        print(f"\n[{name}]  N={len(sub)}  ipsi={ipsi.mean():.4f}  contra={contra.mean():.4f}  "
              f"mean(ipsi-contra)={d.mean():+.4f}")
        print(f"  Wilcoxon paired p={pval:.4f}  | mean asymmetry index={ai.mean():+.3f} "
              f"(neg = ipsilesional reduction)")
        res = {"n": int(len(sub)), "ipsi": float(ipsi.mean()), "contra": float(contra.mean()),
               "delta": float(d.mean()), "p": float(pval), "ai_mean": float(ai.mean())}
        nih = sub["nihss"].values
        if np.isfinite(nih).sum() >= 8:
            rho, pr = spearmanr(nih, ai, nan_policy="omit")
            print(f"  asymmetry vs NIHSS: Spearman rho={rho:+.3f} p={pr:.3f}")
            res["nihss_rho"] = float(rho); res["nihss_p"] = float(pr)
        return res

    out = {"n": n, "n_right": int((df.side == "R").sum()), "n_left": int((df.side == "L").sum()),
           "spindle_sigma_N2": report("SPINDLE (sigma, N2)", "sigma_ipsi", "sigma_contra"),
           "slowwave_delta_N3": report("SLOW-WAVE (delta, N3)", "swa_ipsi", "swa_contra")}
    json.dump(out, open(os.path.join(RES, "lesion_ipsi.json"), "w"), indent=2)
    df.to_csv(os.path.join(RES, "lesion_ipsi_persubject.csv"), index=False)
    print(f"\nsaved -> results/lesion_ipsi.json, results/lesion_ipsi_persubject.csv")
    _figure(df)


def _figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    NAVY = "#1b2a4a"; TEAL = "#2a9d8f"; ORANGE = "#e76f51"
    from scipy.stats import spearmanr
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.0))
    for a, (lab, ic, cc, col) in zip(ax[:2], [
            ("Spindle band (sigma, N2)", "sigma_ipsi", "sigma_contra", TEAL),
            ("Slow-wave band (delta, N3)", "swa_ipsi", "swa_contra", ORANGE)]):
        sub = df.dropna(subset=[ic, cc])
        for _, r in sub.iterrows():
            a.plot([0, 1], [r[ic], r[cc]], color="#bbb", lw=0.8, alpha=0.6, zorder=1)
        a.scatter([0]*len(sub), sub[ic], color=col, s=28, zorder=2, label="ipsilesional")
        a.scatter([1]*len(sub), sub[cc], color=NAVY, s=28, zorder=2, label="contralesional")
        a.plot([0, 1], [sub[ic].mean(), sub[cc].mean()], color="black", lw=2.5, zorder=3, marker="o")
        a.set_xticks([0, 1]); a.set_xticklabels(["Ipsi", "Contra"])
        a.set_title(f"{lab}  (N={len(sub)})", fontsize=12.5, fontweight="bold", color=NAVY)
        a.set_ylabel("Relative power"); a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(fontsize=9, loc="best")
    # panel 3: spindle asymmetry vs NIHSS (the robust finding)
    sub = df.dropna(subset=["sigma_ipsi", "sigma_contra", "nihss"])
    ai = (sub["sigma_ipsi"] - sub["sigma_contra"]) / (sub["sigma_ipsi"] + sub["sigma_contra"])
    rho, pr = spearmanr(sub["nihss"], ai)
    ax[2].scatter(sub["nihss"], ai, color=TEAL, s=36, edgecolor="white", zorder=2)
    if len(sub) >= 3:
        z = np.polyfit(sub["nihss"], ai, 1)
        xs = np.linspace(sub["nihss"].min(), sub["nihss"].max(), 50)
        ax[2].plot(xs, np.polyval(z, xs), color=NAVY, lw=2, zorder=1)
    ax[2].axhline(0, color="#999", ls="--", lw=0.8)
    ax[2].set_xlabel("NIHSS (stroke severity)"); ax[2].set_ylabel("Spindle asymmetry index")
    ax[2].set_title(f"Asymmetry vs severity (N={len(sub)})\nSpearman rho={rho:+.2f}, p={pr:.3f}",
                    fontsize=12.5, fontweight="bold", color=NAVY)
    ax[2].spines[["top", "right"]].set_visible(False)
    fig.suptitle("RQ1/RQ3 preliminary: ipsilesional sleep-microstructure disruption scales with severity",
                 fontsize=14, fontweight="bold", color=NAVY)
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "ipsilesional.png"), bbox_inches="tight"); plt.close()
    print("figure -> presentation/figures/ipsilesional.png")


if __name__ == "__main__":
    main()
