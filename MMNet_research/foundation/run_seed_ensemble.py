"""Seed ensemble of MM-Net on the paper's ten folds, with decisions made on validation patients.

For each fold, the configured model is trained once per seed (42, 1, 7) exactly as
run_10fold trains it (same train/validation split, same HMM fitted on the training
patients). Probabilities are saved for the fold's TEST patients and for its
VALIDATION patients (the ~9 training-side patients run_10fold already holds out for
early stopping). Every choice between decoding variants is made on the validation
patients; the test patients are only scored.

Decoding variants, all followed by the same train-fold HMM:
    single     each seed on its own (reproduces the ordinary single-model numbers)
    ensemble   equal-weight average of the seeds' softmax
    cwcorr     ensemble, then each class probability divided by the class weight the
               loss used (sqrt inverse frequency of the training patients) and
               renormalised -- undoes the weighting's shift toward rare classes. The
               weights come from the training data; nothing is fitted.

Configurations: base and lesion as in run_lesion_aware.py; a "-nocw" suffix trains
with uniform class weights instead of sqrt inverse frequency (otherwise identical).
Importing run_lesion_aware also installs its lean window builder.

Per-fold probabilities: results/revision/runs/lesion/seed_ensemble/<config>_fold<k>.npz
(resumes fold by fold). Summary: seed_ensemble.json / seed_ensemble.txt next to it.

    python run_seed_ensemble.py [--configs base lesion base-nocw] [--seeds 42 1 7]
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

import run_lesion_aware as RL

C, lesion, sweep, arms, FINAL = RL.C, RL.lesion, RL.sweep, RL.arms, RL.FINAL
OUT = os.path.join(RL.OUT, "seed_ensemble")
METRICS = ("acc", "mf1", "kappa")


def split(fi, tr_all):
    """The train/validation split and HMM parameters run_10fold uses for fold fi."""
    rng = np.random.RandomState(100 + fi)
    tr_all = list(tr_all)
    rng.shuffle(tr_all)
    nv = max(10, len(tr_all) // 9)
    va, tr = tr_all[:nv], tr_all[nv:]
    Am = np.ones((C.NC, C.NC))
    pi = np.ones(C.NC)
    for s in tr:
        y = C.DATA[s][2]
        pi[y[0]] += 1
        for x, z in zip(y[:-1], y[1:]):
            Am[x, z] += 1
    return tr, va, np.log(Am / Am.sum(1, keepdims=True)), np.log(pi / pi.sum())


def _fold_ok(path, subs, seeds):
    try:                                   # a power cut can leave a zero-filled file
        with np.load(path) as d:
            return all("SN%d|%d" % (s, sd) in d for s in subs for sd in seeds)
    except Exception:
        return False


def run_config(cfg, seeds):
    os.makedirs(OUT, exist_ok=True)
    # suffixes: -nocw (uniform class weights), -L<n> (temporal window of n epochs),
    # -cross (cross-attention fusion of the neural and cardiorespiratory streams)
    parts = cfg.split("-")
    base_cfg = parts[0] if parts[0] in RL.CONFIGS else cfg
    nocw = "nocw" in parts[1:]
    fusion = "cross" if "cross" in parts[1:] else "concat"
    win = [int(p[1:]) for p in parts[1:] if p.startswith("L") and p[1:].isdigit()]
    steps = RL.CONFIGS[base_cfg]
    orig_cw, orig_L = C.sqrt_cw, C.L
    if nocw:                               # before sweep.config re-compiles train_fold
        C.sqrt_cw = lambda tr: torch.ones(5, dtype=torch.float32, device=C.DEV)
    if win:                                # likewise: the recompiled functions copy L
        C.L = win[0]
    # -ftl: the frozen LaBraM embedding is replaced, fold by fold, by the fine-tuned one
    # (data/cbramod_emb/labramft_f<k>, see build_labramft_dirs.py), so each fold gets its
    # own configuration context
    FTL = {"ftl": "labramft_f%d", "ftlq": "labramftq_f%d", "ftlp": "labramftp_f%d",
           "ftlo": "labramprob_f%d", "ftla": "labramfta_f%d", "ftlap": "labramftap_f%d",
           "ftlae": "labramfta_eog_f%d", "ftlape": "labramftap_eog_f%d"}
    ftl = any(p in FTL for p in parts[1:])
    emb_dir = FTL.get(next((p for p in parts[1:] if p in FTL), "ftl"))
    undo = lesion.install(arms, C, steps) if steps else (lambda: None)

    def context(arm):
        return sweep.config(C, arm, hidden=FINAL["hidden"], drop=FINAL["drop"],
                            lr=FINAL["lr"], wd=FINAL["wd"],
                            cardio=FINAL["cardio"], cardio_mode=FINAL["cardio_mode"])

    def pending(fi):
        tr_all, te = C.FOLDS[fi]
        path = os.path.join(OUT, "%s_fold%d.npz" % (cfg, fi))
        if os.path.exists(path):
            if _fold_ok(path, list(te) + list(split(fi, tr_all)[1]), seeds):
                print("[skip] %s fold %d" % (cfg, fi), flush=True)
                return False
            print("[redo] %s fold %d: saved file incomplete or unreadable" % (cfg, fi), flush=True)
        return True

    def train(fi):
        tr_all, te = C.FOLDS[fi]
        path = os.path.join(OUT, "%s_fold%d.npz" % (cfg, fi))
        tr, va, A_log, pi_log = split(fi, tr_all)
        t0 = time.time()
        probs = {}
        for seed in seeds:
            model = C.train_fold(tr, va, fusion, [], [], seed=seed,
                                 temporal=FINAL["temporal"])
            for s in list(te) + list(va):
                probs["SN%d|%d" % (s, seed)] = C.subj_infer(model, s, [], [])[0]
            del model
            torch.cuda.empty_cache()
        cw = orig_cw(tr).cpu().numpy()    # the weights the published loss uses
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, A_log=A_log, pi_log=pi_log, te=np.array(te),
                            va=np.array(va), seeds=np.array(seeds), cw=cw, **probs)
        os.replace(tmp, path)
        print("[ok] %s fold %d  (%d seeds, %.1f min)"
              % (cfg, fi, len(seeds), (time.time() - t0) / 60), flush=True)

    try:
        if ftl:
            for fi in range(len(C.FOLDS)):
                if pending(fi):
                    with context(FINAL["arm"].replace("labram", emb_dir % fi)):
                        train(fi)
        else:
            with context(FINAL["arm"]):
                for fi in range(len(C.FOLDS)):
                    if pending(fi):
                        train(fi)
    finally:
        undo()
        C.sqrt_cw, C.L = orig_cw, orig_L


def _metrics(yt, yp):
    return dict(acc=float(accuracy_score(yt, yp)),
                mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                kappa=float(cohen_kappa_score(yt, yp)))


def _decode(d, A_log, pi_log, sid, members, cw=None):
    p = np.mean([d[c]["SN%d|%d" % (sid, sd)] for c, sd in members], 0)
    if cw is not None:
        p = p / cw
        p = p / p.sum(1, keepdims=True)
    return C.hmm(A_log, pi_log, np.log(p + C.EPS))


def evaluate(cfgs, seeds):
    """{group: {split: {variant: {metric: [10 fold values]}}}}"""
    groups = {cfg: [(cfg, s) for s in seeds] for cfg in cfgs}
    if len(cfgs) > 1:
        groups["+".join(cfgs)] = [(c, s) for c in cfgs for s in seeds]
    res = {}
    for name, members in groups.items():
        variants = ["%s|%d" % m for m in members] + ["ensemble", "cwcorr"]
        out = {sp: {v: [] for v in variants} for sp in ("va", "te")}
        for fi in range(len(C.FOLDS)):
            d = {c: np.load(os.path.join(OUT, "%s_fold%d.npz" % (c, fi)))
                 for c in {m[0] for m in members}}
            d0 = d[members[0][0]]
            A_log, pi_log, cw = d0["A_log"], d0["pi_log"], d0["cw"]
            for sp in ("va", "te"):
                subs = [int(s) for s in d0[sp]]
                yt = np.concatenate([C.DATA[s][2] for s in subs])
                for m in members:
                    yp = np.concatenate([_decode(d, A_log, pi_log, s, [m]) for s in subs])
                    out[sp]["%s|%d" % m].append(_metrics(yt, yp))
                yp = np.concatenate([_decode(d, A_log, pi_log, s, members) for s in subs])
                out[sp]["ensemble"].append(_metrics(yt, yp))
                yp = np.concatenate([_decode(d, A_log, pi_log, s, members, cw) for s in subs])
                out[sp]["cwcorr"].append(_metrics(yt, yp))
        res[name] = {sp: {v: {k: [f[k] for f in rows] for k in METRICS} for v, rows in o.items()}
                     for sp, o in out.items()}
    return res


def report(res):
    from scipy.stats import wilcoxon
    lines = ["Validation patients (va) drive every choice; test folds (te) are only scored.",
             "%-30s %22s   %22s" % ("", "VALIDATION acc/mf1/kappa", "TEST acc/mf1/kappa")]
    for name, sp in res.items():
        lines.append("== %s" % name)
        for v in sp["te"]:
            lines.append("%-30s " % v + " ".join("%7.4f" % np.mean(sp["va"][v][m]) for m in METRICS)
                         + "   " + " ".join("%7.4f" % np.mean(sp["te"][v][m]) for m in METRICS))
        singles = [v for v in sp["te"] if "|" in v]
        best = max(("ensemble", "cwcorr"), key=lambda v: np.mean(sp["va"][v]["acc"]))
        base = np.mean([sp["te"][v]["acc"] for v in singles], 0)
        for v in ("ensemble", "cwcorr"):
            dd = np.array(sp["te"][v]["acc"]) - base
            p = wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
            lines.append("  TEST %-9s vs mean single seed: acc %+.4f, Wilcoxon p=%.3f, %d/10 folds up"
                         % (v, dd.mean(), p, int((dd > 0).sum())))
        lines.append("  chosen on validation: %s -> TEST acc %.4f"
                     % (best, np.mean(sp["te"][best]["acc"])))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["base", "lesion"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    a = ap.parse_args()
    for cfg in a.configs:
        run_config(cfg, a.seeds)
    res = evaluate(a.configs, a.seeds)
    json.dump(res, open(os.path.join(RL.OUT, "seed_ensemble.json"), "w"), indent=1)
    text = report(res)
    open(os.path.join(RL.OUT, "seed_ensemble.txt"), "w").write(text + "\n")
    print("\n" + text, flush=True)


if __name__ == "__main__":
    main()
