"""The healthy-trained zero-shot models, three seeds each, for a dispersion figure.

The benchmark table gives these rows without an interval because each was trained
once. train_healthy_models.py keys its cache by model name alone and skips a name
it already holds, so re-running it with a different seed does nothing; this
imports its pieces instead of editing it, and keys by model and seed.

One caveat that belongs on the row rather than in a footnote: the dispersion here
is across TRAINING SEEDS, not across folds. The zero-shot evaluation is a single
pass of a finished checkpoint over all 99 patients and has no fold structure, so
this number answers "how much does the checkpoint vary" and not "how much does
the cohort vary", which is what the +- means everywhere else in that table.

  KMP_DUPLICATE_LIB_OK=TRUE python run_zeroshot_seeded.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DG = os.path.join(REPO, "MMNet_research", "domaingap")
sys.path.insert(0, DG)
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import train_healthy_models as T          # noqa: E402

OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final",
                   "zeroshot_seeded.json")
SEEDS = [42, 1, 7]
MODELS = ["cnn", "crnn"]
EPOCHS = 25


def main():
    print("building Sleep-EDF cache ...", flush=True)
    T.build_cache()
    sedf = T.load_sedf()
    keys = sorted(sedf)
    rng = np.random.RandomState(0)
    order = rng.permutation(len(keys))
    n_va = max(2, len(keys) // 5)
    va_keys = [keys[i] for i in order[:n_va]]
    tr_keys = [keys[i] for i in order[n_va:]]
    isl = T.load_isleeps()
    print("sleep-edf %d train / %d val   |   iSLEEPS %d subjects"
          % (len(tr_keys), len(va_keys), len(isl)), flush=True)

    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    t0 = time.time()
    for name in MODELS:
        for seed in SEEDS:
            key = "%s|%d" % (name, seed)
            if key in res:
                print("[skip]", key, flush=True)
                continue
            model, _ = T.train(name, sedf, tr_keys, va_keys, EPOCHS, seed)
            res[key] = dict(healthy=T.evaluate(model, sedf, va_keys),
                            stroke=T.evaluate(model, isl, sorted(isl)),
                            seed=seed)
            json.dump(res, open(OUT, "w"), indent=1)
            print("  %-6s seed %2d   healthy acc %.4f   stroke acc %.4f  [%.1f min]"
                  % (name, seed, res[key]["healthy"]["acc"],
                     res[key]["stroke"]["acc"], (time.time() - t0) / 60), flush=True)

    print("\nZERO-SHOT, dispersion across %d training seeds" % len(SEEDS))
    print("%-6s %-8s %-18s %-18s" % ("model", "domain", "acc", "kappa"))
    summary = {}
    for name in MODELS:
        for dom in ("healthy", "stroke"):
            a = np.array([res["%s|%d" % (name, s)][dom]["acc"] for s in SEEDS])
            k = np.array([res["%s|%d" % (name, s)][dom]["kappa"] for s in SEEDS])
            summary["%s|%s" % (name, dom)] = dict(
                acc=[float(a.mean()), float(a.std(ddof=1))],
                kappa=[float(k.mean()), float(k.std(ddof=1))])
            print("%-6s %-8s %.4f +- %.4f   %.4f +- %.4f"
                  % (name, dom, a.mean(), a.std(ddof=1), k.mean(), k.std(ddof=1)))
    res["_summary"] = summary
    res["_note"] = ("dispersion is across training seeds, not folds; the zero-shot "
                    "evaluation is a single pass over all 99 patients")
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
