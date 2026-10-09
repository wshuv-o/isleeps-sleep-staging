"""Per-fold fine-tuned LaBraM embedding folders, in the frozen cache's format.

For fold k, data/cbramod_emb/labramft_f<k>/SN<n>.npz holds a 200-d embedding per epoch:
    training patients     from the cross-fitted encoders (data/labram_cf/fold<k>/), i.e. an
                          encoder that never saw that patient's labels
    validation + test     from the full-fold encoder (data/labram_ft/fold<k>/), which was
                          trained on the training patients only
so MM-Net, trained on fold k's training patients, never reads an embedding produced by an
encoder fitted on the same patient. arms.py loads these like data/cbramod_emb/labram/.

    python build_labramft_dirs.py [--folds 0 1 ...]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))

import numpy as np

import mmnet_core as C


def split(fi, tr_all):
    """run_10fold's train/validation split for fold fi (same as finetune_labram.split)."""
    rng = np.random.RandomState(100 + fi)
    tr_all = list(tr_all)
    rng.shuffle(tr_all)
    nv = max(10, len(tr_all) // 9)
    return tr_all[nv:], tr_all[:nv]


def _part_of(k, tr, parts=3):
    """Which cross-fitting part held out each training patient (finetune_labram's split)."""
    rng = np.random.RandomState(1000 + k)
    order = sorted(tr)
    rng.shuffle(order)
    return {s: i % parts for i, s in enumerate(order)}


SFX = ""    # "_eog": encoders fine-tuned with E1/E2 (labram_cf_eog, labram_ft_eog)


def _align_maps(k, tr, lam=0.01):
    """Per cross-fitting part, a ridge map from that encoder's embedding space into the
    full-fold encoder's, fitted on the part's own (training) patients only."""
    part = _part_of(k, tr)
    maps = {}
    for p in range(3):
        subs = [s for s in tr if part[s] == p]
        X = np.concatenate([np.load(os.path.join(REPO, "data", "labram_cf" + SFX, "fold%d" % k, "SN%d.npz" % s))["emb"]
                            for s in subs]).astype(np.float64)
        Y = np.concatenate([np.load(os.path.join(REPO, "data", "labram_ft" + SFX, "fold%d" % k, "SN%d.npz" % s))["emb"]
                            for s in subs]).astype(np.float64)
        mx, my = X.mean(0), Y.mean(0)
        Xc, Yc = X - mx, Y - my
        W = np.linalg.solve(Xc.T @ Xc + lam * len(X) * np.eye(X.shape[1]), Xc.T @ Yc)
        maps[p] = (mx, my, W)
    return part, maps


def build(k, quick=False, with_prob=False, prob_only=False, align=False):
    """quick=True: every patient's embedding from the full-fold encoder (training patients
    in-sample) -> labramftq_f<k>. Test embeddings are identical to the clean build; only the
    training patients differ, which makes the quick screen pessimistic, not optimistic."""
    tr_all, te = C.FOLDS[k]
    tr, va = split(k, tr_all)
    name = "labramftq" if quick else "labramft"
    if align:          # cross-fitted embeddings mapped into the full-fold encoder's coordinates
        name += "a"
        part, maps = _align_maps(k, tr)
    if with_prob:      # embedding + the fine-tuned stager's 5 stage probabilities (205-d)
        name += "p"
    if prob_only:      # only the 5 stage probabilities: the same meaning across encoders
        name = "labramprob"
    out = os.path.join(REPO, "data", "cbramod_emb", "%s%s_f%d" % (name, SFX, k))
    os.makedirs(out, exist_ok=True)
    src = {s: ("labram_ft" if quick else "labram_cf") + SFX for s in tr}
    src.update({s: "labram_ft" + SFX for s in list(va) + list(te)})
    for s, d in src.items():
        f = np.load(os.path.join(REPO, "data", d, "fold%d" % k, "SN%d.npz" % s))
        if not np.array_equal(f["y"], C.DATA[s][2]):
            raise ValueError("fold %d SN%d: labels differ from mm_features" % (k, s))
        emb = f["emb"].astype(np.float32)
        if align and d == "labram_cf" + SFX:
            mx, my, W = maps[part[s]]
            emb = ((emb.astype(np.float64) - mx) @ W + my).astype(np.float32)
        if prob_only:
            emb = f["prob"].astype(np.float32)
        elif with_prob:
            emb = np.concatenate([emb, f["prob"].astype(np.float32)], axis=1)
        np.savez_compressed(os.path.join(out, "SN%d.npz" % s), emb=emb.astype(np.float16), y=f["y"])
    print("fold %d: %d training (%s) + %d validation/test (full-fold) -> %s"
          % (k, len(tr), "full-fold, in-sample" if quick else "cross-fitted", len(va) + len(te), out), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--with-prob", action="store_true")
    ap.add_argument("--prob-only", action="store_true")
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--eog", action="store_true", help="use the encoders fine-tuned with E1/E2")
    a = ap.parse_args()
    if a.eog:
        SFX = "_eog"
    for k in a.folds:
        build(k, a.quick, a.with_prob, a.prob_only, a.align)
