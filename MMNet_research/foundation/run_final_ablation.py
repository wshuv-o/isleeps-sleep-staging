"""Modality-removal grid for the final MM-Net (base-nocw-cross-ftlae-p8-al), aligned data.

Same configuration, folds, validation split, seeds, budget and train-fold HMM as the final
model in run_seed_ensemble.py; one modality is removed (zeroed) at training and test time.

Masks for the final model's inputs:
  neural input  = 188 engineered descriptors + 200-d embedding of the fine-tuned EEG+EOG encoder
  cardio input  = 8 raw channels x 750 samples, order ECG, Flow, Thorax, Abdomen, Effort, SpO2,
                  Pulse, nasal pressure (run_mmnet_pflow.table_with_pressure)

  -EEG        EEG descriptors and the whole encoder embedding (the encoder reads the EEG)
  -EOG        EOG descriptors, and the EEG-only fine-tuned aligned encoder (labramfta_f<k>) in
              place of the EEG+EOG one, so no ocular information remains
  -EMG        EMG descriptors (the encoder does not read EMG)
  -SpO2, -pulse/HRV, -ECG    that raw channel
  -airflow    both airflow sensors: oronasal flow and nasal pressure
  -effort     thoracic, abdominal and summed effort
  -all cardio the whole cardiorespiratory tensor

Fold probabilities: results/revision/runs/lesion/seed_ensemble/<config>-x<cond>_fold<k>.npz.
Summary: results/revision/runs/lesion/final_ablation.{json,txt}, three-seed ensemble and
single-model (seed mean) metrics against the joint final model, Holm across the nine removals.

    MMNET_FEATURES=aligned/mm_features MMNET_MULTIMODAL=aligned/multimodal \
    MMNET_PFLOW=aligned/pressure_flow python run_final_ablation.py [--only EEG airflow ...]
"""
import argparse
import json
import os
import sys

import numpy as np

import run_seed_ensemble as SE

C, sweep = SE.C, SE.sweep
FINAL_CFG = "base-nocw-cross-ftlae-p8-al"
EEG_ONLY_CFG = "base-nocw-cross-ftla-p8-al"          # EEG-only fine-tuned aligned encoder
SEEDS = [42, 1, 7]
N_FEAT, N_T = 188, 750
RAW8 = ["ECG", "Flow", "Thorax", "Abdomen", "Effort", "SpO2", "Pulse", "Pressure"]

# table rows first, then the rest of the family
CONDITIONS = {
    "EEG":       dict(eeg=["eeg"], card=[]),
    "allcardio": dict(eeg=[], card=["all"]),
    "effort":    dict(eeg=[], card=["effort"]),
    "airflow":   dict(eeg=[], card=["airflow"]),
    "EOG":       dict(eeg=["eog"], card=[], base=EEG_ONLY_CFG),
    "EMG":       dict(eeg=["emg"], card=[]),
    "SpO2":      dict(eeg=[], card=["spo2"]),
    "pulse":     dict(eeg=[], card=["pulse_hrv"]),
    "ECG":       dict(eeg=[], card=["ecg"]),
}
RAW_GROUPS = {"spo2": ["SpO2"], "pulse_hrv": ["Pulse"], "ecg": ["ECG"],
              "airflow": ["Flow", "Pressure"], "effort": ["Thorax", "Abdomen", "Effort"]}


def masks(n_eeg_total):
    eegm = {}
    for name, m in C.EEGM.items():
        m = np.asarray(m, bool)[:N_FEAT]
        wide = np.zeros(n_eeg_total, bool)
        wide[:N_FEAT] = m
        if name == "eeg":
            wide[N_FEAT:] = True                  # the encoder embedding is EEG-derived
        eegm[name] = wide
    card = {k: [i for c in chans for i in range(RAW8.index(c) * N_T, (RAW8.index(c) + 1) * N_T)]
            for k, chans in RAW_GROUPS.items()}
    return eegm, card


def run_condition(name, spec):
    eeg_drop, card_drop = spec["eeg"], spec["card"]
    base = spec.get("base", FINAL_CFG)
    enter, exit_ = sweep.config.__enter__, sweep.config.__exit__
    saved = {}

    def enter_with_drop(self):
        h = enter(self)
        saved.update(eegm=C.EEGM, card=C.CARD_GROUPS, infer=C.subj_infer)
        C.EEGM, C.CARD_GROUPS = masks(h.dim)
        tf, infer = C.train_fold, C.subj_infer
        C.train_fold = lambda tr, va, fusion, _e, _c, **kw: tf(tr, va, fusion, eeg_drop, card_drop, **kw)
        C.subj_infer = lambda model, s, _e, _c: infer(model, s, eeg_drop, card_drop)
        print("[%s] neural dim %d, eeg_drop %s (%d dims), card_drop %s (%s samples)"
              % (name, h.dim, eeg_drop, sum(int(C.EEGM[m].sum()) for m in eeg_drop), card_drop,
                 "all" if "all" in card_drop else sum(len(C.CARD_GROUPS[g]) for g in card_drop)),
              flush=True)
        return h

    def exit_restore(self, *a):
        C.EEGM, C.CARD_GROUPS, C.subj_infer = saved["eegm"], saved["card"], saved["infer"]
        return exit_(self, *a)

    sweep.config.__enter__, sweep.config.__exit__ = enter_with_drop, exit_restore
    try:
        SE.run_config("%s-x%s" % (base, name), SEEDS)
    finally:
        sweep.config.__enter__, sweep.config.__exit__ = enter, exit_


def fold_metrics(cfg):
    """Per fold: three-seed ensemble and seed-mean staging accuracy, and respiratory AUC."""
    from sklearn.metrics import roc_auc_score
    out = {"acc_ens": [], "acc_single": [], "auc_ens": [], "auc_single": []}
    for fi in range(len(C.FOLDS)):
        d = np.load(os.path.join(SE.OUT, "%s_fold%d.npz" % (cfg, fi)))
        subs = [int(s) for s in d["te"]]
        yt = np.concatenate([C.DATA[s][2] for s in subs])
        ya = np.concatenate([C.DATA[s][3] for s in subs]).astype(int)
        dec = lambda members: np.concatenate([SE._decode({cfg: d}, d["A_log"], d["pi_log"], s, members)
                                              for s in subs])
        out["acc_ens"].append(float((dec([(cfg, sd) for sd in SEEDS]) == yt).mean()))
        out["acc_single"].append(float(np.mean([(dec([(cfg, sd)]) == yt).mean() for sd in SEEDS])))
        ps = [np.concatenate([d["A:SN%d|%d" % (s, sd)] for s in subs]) for sd in SEEDS]
        out["auc_ens"].append(float(roc_auc_score(ya, np.mean(ps, 0))))
        out["auc_single"].append(float(np.mean([roc_auc_score(ya, p) for p in ps])))
    return out


def summarize(names):
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests
    ref = fold_metrics(FINAL_CFG)
    res = {"reference": ref}
    rows = []
    for n in names:
        spec = CONDITIONS[n]
        m = fold_metrics("%s-x%s" % (spec.get("base", FINAL_CFG), n))
        res[n] = m
        rows.append((n, m))
    lines = ["Final MM-Net (%s), modality removal, aligned data, seeds %s" % (FINAL_CFG, SEEDS),
             "reference: 3-seed acc %.4f AUC %.4f | single acc %.4f AUC %.4f"
             % tuple(np.mean(ref[k]) for k in ("acc_ens", "auc_ens", "acc_single", "auc_single"))]
    for key, label in (("ens", "three-seed ensemble"), ("single", "single model (seed mean)")):
        pa = [wilcoxon(ref["acc_" + key], m["acc_" + key]).pvalue for _, m in rows]
        pu = [wilcoxon(ref["auc_" + key], m["auc_" + key]).pvalue for _, m in rows]
        ha = multipletests(pa, method="holm")[1]
        hu = multipletests(pu, method="holm")[1]
        lines.append("\n%s (Holm across %d removals)" % (label, len(rows)))
        for (n, m), a, u in zip(rows, ha, hu):
            lines.append("  -%-10s acc %.4f (%+.4f, Holm p=%.3f)   AUC %.4f (%+.4f, Holm p=%.3f)"
                         % (n, np.mean(m["acc_" + key]), np.mean(m["acc_" + key]) - np.mean(ref["acc_" + key]), a,
                            np.mean(m["auc_" + key]), np.mean(m["auc_" + key]) - np.mean(ref["auc_" + key]), u))
            res[n]["holm_%s" % key] = {"acc": float(a), "auc": float(u)}
    return res, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS))
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if not a.summary_only:
        for n in a.only:
            run_condition(n, CONDITIONS[n])
    done = [n for n in CONDITIONS
            if all(os.path.exists(os.path.join(SE.OUT, "%s-x%s_fold%d.npz"
                                               % (CONDITIONS[n].get("base", FINAL_CFG), n, fi)))
                   for fi in range(len(C.FOLDS)))]
    res, text = summarize(done)
    out = os.path.join(SE.RL.OUT, "final_ablation")
    json.dump(res, open(out + ".json", "w"), indent=1)
    open(out + ".txt", "w").write(text + "\n")
    print("\n" + text, flush=True)


if __name__ == "__main__":
    sys.exit(main())
