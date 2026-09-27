"""Statistical re-analyses from stored fold-level results (CPU only, seconds).

Referee items covered:
  6D  healthy -> stroke drop for every model with both scores
  6J  the two dispersion estimators (30 fold-values vs 10 fold-means)
  6M  fold-to-fold vs seed-to-seed spread
  6Q  Holm across all nine same-fold baselines including Huttunen et al.
  6H  what the pretrained embedding adds inside the adopted (BiLSTM + CNN) model
  7h  learning curve: paired test of the last step
  9a  Benjamini-Hochberg over the secondary tests reported uncorrected
  9b  Nadeau-Bengio corrected resampled t-test for the main paired comparisons
  9c  staging accuracy vs AHI: trend test, and the same with wake burden partialled out
  3   nested respiratory estimate (restated from nested_cv.json)
  7e  the median-accuracy patient, as the candidate for Fig. 3

  python stats_round3.py
"""
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr, wilcoxon, t as tdist, norm

import _common as C

F = C.FINAL
RUNS = os.path.dirname(F)
BENCH = os.path.join(C.MM, "results", "benchmark")


def J(name, base=F):
    return json.load(open(os.path.join(base, name)))


def fm(seed_lists):
    """fold-means over seeds: list of per-seed lists of 10 -> array(10)"""
    return np.mean(np.array(seed_lists, float), axis=0)


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i]))
        adj[i] = run
    return adj


def bh(ps):
    ps = np.asarray(ps, float)
    m = len(ps)
    o = np.argsort(ps)[::-1]
    adj = np.empty(m)
    prev = 1.0
    for r, i in enumerate(o):
        k = m - r
        prev = min(prev, ps[i] * m / k)
        adj[i] = prev
    return adj


def nadeau_bengio(a, b, test_frac=0.1):
    """Corrected resampled t (Nadeau & Bengio 2003) on K paired fold values."""
    d = np.asarray(a, float) - np.asarray(b, float)
    k = len(d)
    ratio = test_frac / (1 - test_frac)
    var = (1.0 / k + ratio) * d.var(ddof=1)
    t = d.mean() / np.sqrt(var) if var > 0 else np.inf
    p = 2 * tdist.sf(abs(t), k - 1)
    tn = d.mean() / np.sqrt(d.var(ddof=1) / k)
    return dict(diff=float(d.mean()), t_corrected=float(t), p_corrected=float(p),
                t_naive=float(tn), p_naive_t=float(2 * tdist.sf(abs(tn), k - 1)),
                p_wilcoxon=float(wilcoxon(a, b).pvalue), wins=int((d > 0).sum()))


def main():
    out = {}
    final = J("final_model.json")
    S = ["final|42", "final|1", "final|7"]
    full_acc = fm([final[s]["acc"] for s in S])
    full_auc = fm([final[s]["auc"] for s in S])
    full_acc42 = np.array(final["final|42"]["acc"])

    # ---- 6J / 6M dispersion -------------------------------------------------
    A = np.array([final[s]["acc"] for s in S])
    U = np.array([final[s]["auc"] for s in S])
    out["dispersion"] = {
        "acc_sd_30_values": float(A.std(ddof=1)), "acc_sd_10_fold_means": float(A.mean(0).std(ddof=1)),
        "auc_sd_30_values": float(U.std(ddof=1)), "auc_sd_10_fold_means": float(U.mean(0).std(ddof=1)),
        "acc_seed_means": A.mean(1).tolist(), "auc_seed_means": U.mean(1).tolist(),
        "acc_sd_across_seed_means": float(A.mean(1).std(ddof=1)),
        "auc_sd_across_seed_means": float(U.mean(1).std(ddof=1)),
        "ratio_fold_to_seed_acc_30": float(A.std(ddof=1) / A.mean(1).std(ddof=1)),
        "ratio_fold_to_seed_auc_30": float(U.std(ddof=1) / U.mean(1).std(ddof=1)),
        "ratio_fold_to_seed_auc_10": float(U.mean(0).std(ddof=1) / U.mean(1).std(ddof=1)),
        "table11_adopted_row_is_separate_run": J("experiment_table.json", os.path.join(RUNS, "foundation"))[3],
    }

    # ---- 6D healthy -> stroke ------------------------------------------------
    def m(d, k="acc"):
        v = d[k]
        return v["mean"] if isinstance(v, dict) else float(np.mean(v))
    att = J("attnsleep_seeded.json")
    att_st = float(np.mean([np.mean(att["per_seed"][s]["acc"]) for s in att["per_seed"]]))
    no = J("neural_only_pcf.json")
    no_acc = fm([no["neural_only|%d" % s]["acc"] for s in (42, 1, 7)])
    ds = J("deepsleep_all.json", BENCH)
    cn = J("cnn4ch_all.json", BENCH)
    rows = [
        ("U-Time", m(J("utime_healthy.json")), m(J("utime_stroke.json")), "10-fold, 99"),
        ("TinySleepNet", m(J("healthy_tiny.json")), m(J("tinysleepnet.json")), "10-fold, 99"),
        ("SleepTransformer", m(J("healthy_transformer.json")), m(J("sleeptransformer.json")), "10-fold, 99"),
        ("1D-ResNet-SE-LSTM", m(J("resnetse_healthy.json")), m(J("resnetse_stroke.json")), "10-fold, 99"),
        ("Micro SleepNet", m(J("micro_healthy.json")), m(J("micro_stroke.json")), "10-fold, 99"),
        ("AT-BiLSTM", m(J("atbilstm_healthy.json")), m(J("atbilstm_stroke.json")), "10-fold, 99"),
        ("AttnSleep", m(J("healthy_attnsleep.json")), att_st, "10-fold x 3 seeds, 99"),
        ("DeepSleepNet", m(J("healthy_deepsleep.json")), float(np.mean([f["acc"] for f in ds["folds"]])),
         "stroke row: 5 folds, %d epochs (not the 99-patient folds)" % sum(f["n"] for f in ds["folds"])),
        ("CNN+BiLSTM (4 EEG)", m(J("healthy_cnn.json")), float(np.mean([f["acc"] for f in cn["folds"]])),
         "stroke row: 5 folds, %d epochs (not the 99-patient folds)" % sum(f["n"] for f in cn["folds"])),
        ("MM-Net (neural only)", J("healthy_cv.json")["acc"]["mean"], float(no_acc.mean()), "10-fold x 3 seeds, 99"),
    ]
    out["healthy_to_stroke"] = [dict(model=r[0], healthy=r[1], stroke=r[2], drop=r[1] - r[2], note=r[3]) for r in rows]
    drops = [r[1] - r[2] for r in rows[:-1]]
    out["healthy_to_stroke_range_baselines"] = [float(min(drops)), float(max(drops))]

    # ---- 6Q / 9b baselines on the same ten folds ----------------------------
    def perfold(d):
        return [f["acc"] for f in d["per_fold"]]
    hut = J("huttunen.json")
    base = {
        "AttnSleep": fm([att["per_seed"][s]["acc"] for s in att["per_seed"]]),
        "LSTM reimplementation": fm([J("isleeps_lstm.json")["per_seed"][s]["acc"] for s in J("isleeps_lstm.json")["per_seed"]]),
        "1D-ResNet-SE-LSTM": np.array(perfold(J("resnetse_stroke.json"))),
        "AT-BiLSTM": np.array(perfold(J("atbilstm_stroke.json"))),
        "TinySleepNet": np.array(perfold(J("tinysleepnet.json"))),
        "SleepTransformer": np.array(perfold(J("sleeptransformer.json"))),
        "U-Time": np.array(perfold(J("utime_stroke.json"))),
        "Micro SleepNet": np.array(perfold(J("micro_stroke.json"))),
        "Huttunen et al. (port)": fm([hut["seed|%d" % s]["acc"] for s in (42, 1, 7)]),
    }
    seeded = {"AttnSleep", "LSTM reimplementation", "Huttunen et al. (port)"}
    # primary: MM-Net 3-seed fold means against every baseline (the paper's reference);
    # sensitivity: MM-Net seed 42 against the single-seed baselines
    res, res42 = {}, {}
    for k, v in base.items():
        res[k] = nadeau_bengio(full_acc, v)
        res42[k] = nadeau_bengio(full_acc if k in seeded else full_acc42, v)
    for R in (res, res42):
        names = list(R)
        hw = holm([R[k]["p_wilcoxon"] for k in names])
        hnb = holm([R[k]["p_corrected"] for k in names])
        for k, a, b in zip(names, hw, hnb):
            R[k]["holm_wilcoxon_9"] = float(a)
            R[k]["holm_nadeau_bengio_9"] = float(b)
    out["baselines_accuracy_paired"] = res
    out["baselines_accuracy_paired_seed42_reference"] = res42

    # ---- 9b main internal comparisons ---------------------------------------
    def shard(cond):
        v = []
        for s in (42, 1, 7):
            p = os.path.join(F, "ablation_shards", "%s__s%d.json" % (cond, s))
            if os.path.exists(p):
                d = json.load(open(p))
                v.append(list(d.values())[0])
        return v
    comps = {}
    comps["neural-only vs full, accuracy"] = nadeau_bengio(no_acc, full_acc)
    no_auc = fm([no["neural_only|%d" % s]["auc"] for s in (42, 1, 7)])
    comps["full vs neural-only, respiratory AUC"] = nadeau_bengio(full_auc, no_auc)
    for cond, key, lab in (("noall_cardio", "auc", "full vs -all cardiorespiratory, AUC"),
                           ("noeffort", "auc", "full vs -effort, AUC"),
                           ("noEEG", "acc", "full vs -EEG, accuracy"),
                           ("noairflow", "auc", "full vs -airflow, AUC")):
        v = shard(cond)
        if len(v) == 3:
            comps[lab] = nadeau_bengio(full_acc if key == "acc" else full_auc, fm([x[key] for x in v]))
    bp = J("bypass_ablation.json")["per_seed"]["no_bypass"]
    comps["bypass vs zeroed, AUC"] = nadeau_bengio(full_auc, fm([bp[s]["auc"] for s in bp]))
    st = J("single_task.json")
    comps["joint vs staging-only, accuracy"] = nadeau_bengio(full_acc, fm([st["stage|%d" % s]["acc"] for s in (42, 1, 7)]))
    comps["joint vs respiratory-only, AUC"] = nadeau_bengio(full_auc, fm([st["apnea|%d" % s]["auc"] for s in (42, 1, 7)]))
    comps["Huttunen port vs MM-Net, AUC"] = nadeau_bengio(full_auc, fm([hut["seed|%d" % s]["auc"] for s in (42, 1, 7)]))
    # 6H: pretrained embedding inside the adopted model (sweep shards, same config +/- LaBraM)
    sw = os.path.join(RUNS, "foundation", "sweep_shards")
    def sweep(arm):
        v = []
        for s in (42, 1, 7):
            d = json.load(open(os.path.join(sw, "%s__tlstm__h256__dr0.3__lr0.0003__wd0.0001__s%d__craw_cnn-concat.json" % (arm, s))))
            v.append(list(d.values())[0])
        return v
    wl, nl = sweep("A+labram"), sweep("A")
    comps["LaBraM in adopted model (A+labram vs A, both BiLSTM+CNN), accuracy"] = nadeau_bengio(fm([x["acc"] for x in wl]), fm([x["acc"] for x in nl]))
    comps["LaBraM in adopted model, AUC"] = nadeau_bengio(fm([x["auc"] for x in wl]), fm([x["auc"] for x in nl]))
    comps["_means"] = {"A+labram_raw_cnn_acc": float(fm([x["acc"] for x in wl]).mean()),
                       "A_raw_cnn_acc": float(fm([x["acc"] for x in nl]).mean()),
                       "A+labram_raw_cnn_auc": float(fm([x["auc"] for x in wl]).mean()),
                       "A_raw_cnn_auc": float(fm([x["auc"] for x in nl]).mean())}
    out["nadeau_bengio_internal"] = comps

    # ---- 7h learning curve ---------------------------------------------------
    lc = {}
    for f in sorted(glob.glob(os.path.join(F, "lc_shards", "*.json"))):
        v = list(json.load(open(f)).values())
        lc[os.path.basename(f)] = dict(n_train=float(np.mean([x["n_train"] for x in v])),
                                       acc=[x["acc"] for x in v], auc=[x["auc"] for x in v])
    k = sorted(lc)
    summary = [dict(file=x, n_train=lc[x]["n_train"], acc_mean=float(np.mean(lc[x]["acc"])),
                    acc_sd=float(np.std(lc[x]["acc"], ddof=1)), auc_mean=float(np.mean(lc[x]["auc"])),
                    auc_sd=float(np.std(lc[x]["auc"], ddof=1))) for x in k]
    steps = []
    for a, b in zip(k[:-1], k[1:]):
        steps.append(dict(step="%s->%s" % (lc[a]["n_train"], lc[b]["n_train"]),
                          acc=nadeau_bengio(lc[b]["acc"], lc[a]["acc"]),
                          auc=nadeau_bengio(lc[b]["auc"], lc[a]["auc"])))
    out["learning_curve"] = dict(points=summary, paired_steps=steps,
                                 note="single seed (42); nested training subsets, same test folds")

    # ---- 9c severity trend ----------------------------------------------------
    P, raw = C.probs()
    A_ = C.ahi()
    ids = [s for s in C.SUBJECTS if s in A_]
    acc = np.array([raw[s]["acc"] for s in ids])
    ahi = np.array([A_[s] for s in ids])
    wake = np.array([(C.stages(s) == 0).mean() for s in ids])
    r = spearmanr(acc, ahi)
    # Jonckheere-Terpstra across AASM bands, normal approximation
    band = np.digitize(ahi, [5, 15, 30])
    grp = [acc[band == g] for g in range(4)]
    J_ = sum(((gj[:, None] < gk[None, :]).sum() + 0.5 * (gj[:, None] == gk[None, :]).sum())
             for i, gj in enumerate(grp) for gk in grp[i + 1:])
    n = np.array([len(g) for g in grp]); N = n.sum()
    mu = (N ** 2 - (n ** 2).sum()) / 4.0
    var = (N ** 2 * (2 * N + 3) - (n ** 2 * (2 * n + 3)).sum()) / 72.0
    z = (J_ - mu) / np.sqrt(var)
    # partial Spearman: residualise ranks on wake fraction
    from scipy.stats import rankdata
    ra, rh, rw = rankdata(acc), rankdata(ahi), rankdata(wake)
    res_a = ra - np.polyval(np.polyfit(rw, ra, 1), rw)
    res_h = rh - np.polyval(np.polyfit(rw, rh, 1), rw)
    pr = np.corrcoef(res_a, res_h)[0, 1]
    tp = pr * np.sqrt((len(ids) - 3) / (1 - pr ** 2))
    pp = 2 * tdist.sf(abs(tp), len(ids) - 3)
    mod, sev = acc[band == 2], acc[band == 3]
    from scipy.stats import mannwhitneyu
    out["severity"] = dict(
        n=len(ids), band_n=n.tolist(), band_mean_acc=[float(g.mean()) for g in grp],
        spearman_acc_vs_ahi=float(r.statistic), p=float(r.pvalue),
        jonckheere_z_decreasing=float(-z), jonckheere_p_one_sided=float(norm.sf(-z)),
        rho_wake_vs_ahi=float(spearmanr(wake, ahi).statistic),
        rho_wake_vs_acc=float(spearmanr(wake, acc).statistic),
        partial_spearman_acc_ahi_given_wake=float(pr), partial_p=float(pp),
        moderate_vs_severe_mannwhitney_p=float(mannwhitneyu(mod, sev).pvalue))

    # ---- 9a Benjamini-Hochberg over secondary tests (reported p-values) -------
    sec = {
        "bypass vs zeroed, AUC (Wilcoxon, fold means)": J("bypass_matched.json")["_summary"]["auc"]["p_vs_zeroed"],
        "bypass vs shuffled, AUC": J("bypass_matched.json")["_summary"]["auc"]["p_vs_shuffled"],
        "joint vs staging-only, accuracy": comps["joint vs staging-only, accuracy"]["p_wilcoxon"],
        "joint vs respiratory-only, AUC": comps["joint vs respiratory-only, AUC"]["p_wilcoxon"],
        "attention vs concat, AUC (paper, feature-only model)": 0.06,
        "attention vs concat, accuracy (paper)": 0.49,
        "incomplete montage: full model staging gap (paper)": 0.017,
        "incomplete montage: neural-only staging gap (paper)": 0.040,
        "incomplete montage: full vs neural-only on 12 (paper)": 0.18,
        "Huttunen port vs MM-Net, accuracy": res["Huttunen et al. (port)"]["p_wilcoxon"],
        "N1 F1 gain from cardio stream (fold_level_tests.json)": J("fold_level_tests.json")["n1_gain"]["p10"],
        "transductive vs training-population normalisation, accuracy (paper)": 0.28,
    }
    ks = list(sec)
    adj = bh([sec[x] for x in ks])
    adjh = holm([sec[x] for x in ks])
    out["secondary_tests"] = {x: dict(p=float(sec[x]), bh=float(a), holm=float(h)) for x, a, h in zip(ks, adj, adjh)}

    # ---- 3 nested --------------------------------------------------------------
    nc = J("nested_cv.json")["outer"]
    out["nested"] = {m_: dict(mean=float(np.mean([nc[o][m_] for o in nc])), sd=float(np.std([nc[o][m_] for o in nc], ddof=1)))
                     for m_ in ("acc", "mf1", "kappa", "auc", "ap")}
    out["nested"]["selected"] = [nc[o]["selected"].split("c=")[-1] for o in sorted(nc)]
    out["nested"]["reported"] = dict(acc=float(full_acc.mean()), auc=float(full_auc.mean()))

    # ---- 7e median patient -------------------------------------------------------
    accs = sorted((raw[s]["acc"], s) for s in raw)
    out["fig3_candidates"] = dict(median=accs[len(accs) // 2], sn90=raw["SN90"]["acc"],
                                  mean=float(np.mean([a for a, _ in accs])))
    C.save("stats_round3.json", out)
    print(json.dumps({k: out[k] for k in ("healthy_to_stroke_range_baselines", "severity", "nested", "fig3_candidates")}, indent=1, default=float))
    for r_ in out["healthy_to_stroke"]:
        print("%-24s %.3f -> %.3f  drop %.3f  %s" % (r_["model"], r_["healthy"], r_["stroke"], r_["drop"], r_["note"]))
    for kk, v in res.items():
        print("%-24s diff %+.3f wins %d  wil %.4f holm %.3f | NB p %.4f holm %.3f" % (kk, v["diff"], v["wins"], v["p_wilcoxon"], v["holm_wilcoxon_9"], v["p_corrected"], v["holm_nadeau_bengio_9"]))
    for kk, v in comps.items():
        if kk.startswith("_"):
            print(kk, v); continue
        print("%-60s diff %+.4f wins %d wil %.4f NB %.4f naive-t %.4f" % (kk, v["diff"], v["wins"], v["p_wilcoxon"], v["p_corrected"], v["p_naive_t"]))
    for s_ in steps:
        print(s_["step"], "acc %+.3f p %.3f/%.3f" % (s_["acc"]["diff"], s_["acc"]["p_wilcoxon"], s_["acc"]["p_corrected"]),
              "auc %+.3f wins %d p %.3f/%.3f" % (s_["auc"]["diff"], s_["auc"]["wins"], s_["auc"]["p_wilcoxon"], s_["auc"]["p_corrected"]))
    for kk, v in out["secondary_tests"].items():
        print("%-70s p %.3f  BH %.3f  Holm %.3f" % (kk, v["p"], v["bh"], v["holm"]))
    print(out["dispersion"])


if __name__ == "__main__":
    main()
