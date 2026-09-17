"""Nested cross-validation: select the architecture inside the training data only.

Both reviews call this the paper's largest methodological vulnerability, and they
are right. The reported configuration was chosen by comparing forty-nine candidates
on the same ten folds that produce the headline number, so those figures carry an
unknown amount of selection optimism. The paper bounds it and says so, but a bound
is not a removal.

This removes it. The design is the one the review proposed: five outer
patient-independent folds, and inside each outer training set three inner folds
over which all forty-nine candidates are compared. The winner of the inner
comparison is retrained on the whole outer training set and scored once on the
outer test patients, who took no part in selecting it. The outer estimate is
therefore of the whole procedure -- search included -- rather than of a
configuration that already knew its test set.

The candidate list is not invented here. It is read back from the sweep's own
shard files, which carry exactly forty-nine distinct configurations once the seed
is stripped, so the space searched inside each fold is the space the paper says
was searched.

Two honest limits. Five outer folds are fewer than the paper's ten, so the outer
estimate is noisier than the in-domain figures it is compared against; and a single
seed is used throughout, because 5 x 49 x 3 trainings is already the better part of
a day. Neither affects the point, which is whether the number moves once selection
is quarantined.

Resumable at the granularity of one (outer, inner, candidate) evaluation, and a
candidate that raises is recorded and skipped rather than killing the run.

  KMP_DUPLICATE_LIB_OK=TRUE python run_nested_cv.py
"""
import glob
import json
import os
import re
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import mmnet_core as C          # noqa: E402
import sweep                    # noqa: E402

OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final")
SHARDS = os.path.join(REPO, "MMNet_research", "results", "revision", "runs",
                      "foundation", "sweep_shards")
PATH = os.path.join(OUT, "nested_cv.json")
N_OUTER, N_INNER, SEED = 5, 3, 42


def candidates():
    """The forty-nine configurations, read back from the sweep's own shards."""
    keys = set()
    for f in glob.glob(os.path.join(SHARDS, "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if isinstance(d, dict):
            keys |= {k for k in d if isinstance(k, str) and not k.startswith("_")}
    return sorted({re.sub(r"\|s\d+", "", k) for k in keys})


def parse(label):
    """'A+labram|t=lstm|h256|dr0.3|lr0.0003|wd0.0001|c=raw_cnn-concat' -> kwargs."""
    parts = label.split("|")
    cfg = dict(arm=parts[0], temporal="none", hidden=128, drop=0.3,
               lr=1e-3, wd=1e-4, cardio=None, cardio_mode="concat")
    for p in parts[1:]:
        if p.startswith("t="):
            cfg["temporal"] = p[2:]
        elif p.startswith("h"):
            cfg["hidden"] = int(p[1:])
        elif p.startswith("dr"):
            cfg["drop"] = float(p[2:])
        elif p.startswith("lr"):
            cfg["lr"] = float(p[2:])
        elif p.startswith("wd"):
            cfg["wd"] = float(p[2:])
        elif p.startswith("c="):
            name, _, mode = p[2:].partition("-")
            cfg["cardio"], cfg["cardio_mode"] = name, mode or "concat"
    return cfg


def folds_of(subjects, k, seed):
    rng = np.random.RandomState(seed)
    s = list(subjects)
    rng.shuffle(s)
    return [sorted(s[i::k]) for i in range(k)]


def train_eval(cfg, tr, va, te, seed):
    """Train one candidate and return its accuracy on `te`."""
    with sweep.config(C, cfg["arm"], hidden=cfg["hidden"], drop=cfg["drop"],
                      lr=cfg["lr"], wd=cfg["wd"], cardio=cfg["cardio"],
                      cardio_mode=cfg["cardio_mode"]):
        model = C.train_fold(tr, va, "concat", [], [], seed=seed,
                             temporal=cfg["temporal"])
        acc, n = 0.0, 0
        for s in te:
            sp, _ = C.subj_infer(model, s, [], [])
            y = C.DATA[s][2]
            acc += (sp.argmax(1) == y).sum()
            n += len(y)
    return float(acc / max(1, n))


def scored(cfg, tr, va, te, seed):
    """Full metrics for the selected candidate on the outer test fold."""
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 cohen_kappa_score, f1_score, roc_auc_score)
    with sweep.config(C, cfg["arm"], hidden=cfg["hidden"], drop=cfg["drop"],
                      lr=cfg["lr"], wd=cfg["wd"], cardio=cfg["cardio"],
                      cardio_mode=cfg["cardio_mode"]):
        model = C.train_fold(tr, va, "concat", [], [], seed=seed,
                             temporal=cfg["temporal"])
        Am = np.ones((C.NC, C.NC)); pi = np.ones(C.NC)
        for s in tr:
            y = C.DATA[s][2]; pi[y[0]] += 1
            for x, z in zip(y[:-1], y[1:]):
                Am[x, z] += 1
        A_log = np.log(Am / Am.sum(1, keepdims=True))
        pi_log = np.log(pi / pi.sum())
        yt, yp, at, asc = [], [], [], []
        for s in te:
            sp, apn = C.subj_infer(model, s, [], [])
            yt.append(C.DATA[s][2]); yp.append(C.hmm(A_log, pi_log, np.log(sp + C.EPS)))
            at.append(C.DATA[s][3]); asc.append(apn)
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    at, asc = np.concatenate(at), np.concatenate(asc)
    return dict(acc=float(accuracy_score(yt, yp)),
                mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                kappa=float(cohen_kappa_score(yt, yp)),
                auc=float(roc_auc_score(at, asc)) if len(np.unique(at)) > 1 else float("nan"),
                ap=float(average_precision_score(at, asc)) if len(np.unique(at)) > 1 else float("nan"))


def main():
    os.makedirs(OUT, exist_ok=True)
    res = json.load(open(PATH)) if os.path.exists(PATH) else {"inner": {}, "outer": {}, "failed": {}}
    cands = candidates()
    print("candidates recovered from the sweep shards: %d" % len(cands), flush=True)
    if len(cands) != 49:
        print("  WARNING: expected 49; the shard set has changed", flush=True)

    outer = folds_of(C.SUBS, N_OUTER, SEED)
    t0 = time.time()
    total = N_OUTER * N_INNER * len(cands)
    done = len(res["inner"])

    for oi, te_o in enumerate(outer):
        tr_o = [s for s in C.SUBS if s not in te_o]
        inner = folds_of(tr_o, N_INNER, SEED + 100 + oi)
        for ii, va_i in enumerate(inner):
            tr_i = [s for s in tr_o if s not in va_i]
            hold = max(10, len(tr_i) // 9)
            rng = np.random.RandomState(200 + oi * 10 + ii)
            t = list(tr_i); rng.shuffle(t)
            es, fit = t[:hold], t[hold:]          # early-stopping patients
            for label in cands:
                key = "o%d|i%d|%s" % (oi, ii, label)
                if key in res["inner"] or key in res["failed"]:
                    continue
                try:
                    a = train_eval(parse(label), fit, es, va_i, SEED)
                    res["inner"][key] = a
                except (MemoryError, RuntimeError) as e:
                    # Running out of memory says nothing about the candidate. Recording
                    # it as a failure would skip it on every later run, which is how a
                    # transient collision turned into forty-nine permanently missing
                    # evaluations the first time this ran.
                    print("  transient on %s: %s -- will retry on the next pass"
                          % (key, type(e).__name__), flush=True)
                except Exception as e:
                    res["failed"][key] = "%s: %s" % (type(e).__name__, e)
                    print("  FAILED %s -> %s" % (key, res["failed"][key]), flush=True)
                    traceback.print_exc()
                done = len(res["inner"]) + len(res["failed"])
                json.dump(res, open(PATH, "w"), indent=1)
                if done % 10 == 0:
                    el = (time.time() - t0) / 60
                    print("  %d/%d inner evaluations, %.0f min elapsed" % (done, total, el),
                          flush=True)

        # ---- select on the inner folds, score once on the outer test --------
        okey = "o%d" % oi
        if okey in res["outer"]:
            print("[skip] outer %d" % oi, flush=True)
            continue
        means = {}
        for label in cands:
            vals = [res["inner"].get("o%d|i%d|%s" % (oi, ii, label)) for ii in range(N_INNER)]
            vals = [v for v in vals if v is not None]
            if vals:
                means[label] = float(np.mean(vals))
        if not means:
            print("outer %d: no candidate survived; skipping" % oi, flush=True)
            continue
        best = max(means, key=means.get)
        hold = max(10, len(tr_o) // 9)
        rng = np.random.RandomState(300 + oi)
        t = list(tr_o); rng.shuffle(t)
        es, fit = t[:hold], t[hold:]
        m = scored(parse(best), fit, es, te_o, SEED)
        res["outer"][okey] = dict(selected=best, inner_acc=round(means[best], 4),
                                  n_candidates=len(means), **m)
        json.dump(res, open(PATH, "w"), indent=1)
        print("outer %d: selected %s (inner %.4f) -> outer acc %.4f kappa %.4f auc %.4f  [%.0f min]"
              % (oi, best, means[best], m["acc"], m["kappa"], m["auc"],
                 (time.time() - t0) / 60), flush=True)

    # ---- the comparison the review asked for -------------------------------
    outs = [res["outer"]["o%d" % i] for i in range(N_OUTER) if "o%d" % i in res["outer"]]
    if outs:
        fm = json.load(open(os.path.join(OUT, "final_model.json")))
        pub = float(np.mean([np.mean(fm["final|%d" % s]["acc"]) for s in (42, 1, 7)]))
        acc = np.array([o["acc"] for o in outs])
        res["_summary"] = dict(
            nested_acc=round(float(acc.mean()), 4), nested_sd=round(float(acc.std(ddof=1)), 4),
            published_acc=round(pub, 4), optimism=round(float(pub - acc.mean()), 4),
            selected=[o["selected"] for o in outs], n_outer=len(outs))
        print("\nnested %.4f +/- %.4f over %d outer folds; published %.4f; difference %+.4f"
              % (acc.mean(), acc.std(ddof=1), len(outs), pub, pub - acc.mean()))
        print("selected per outer fold:")
        for i, o in enumerate(outs):
            print("  o%d  %s" % (i, o["selected"]))
        json.dump(res, open(PATH, "w"), indent=1)
    print("\nsaved -> %s  [%.0f min]" % (PATH, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
