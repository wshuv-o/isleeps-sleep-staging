"""A capacity-matched control for the direct cardiorespiratory path.

The review asks whether the bypass gain is an information effect or a capacity
effect: a head that reads 576 inputs has more first-layer weights than one reading
512, and the ordinary ablation might be measuring that instead of the physiology.

For this implementation the capacity question is already answered. mmnet_core
builds apnea_head as Linear(2*hidden + d_card, hidden) regardless of the flag, and
bypass=False zeroes the cardio block rather than removing it, so the with- and
without-bypass models have identical parameter counts. The script asserts that
rather than asserting it in prose.

What is not yet answered is the sharper version of the question. Zeros are not a
neutral input; a head fed a constant learns to ignore that block, so the ablation
compares "aligned cardio" against "nothing" and cannot separate aligned information
from the mere presence of a plausible signal. This adds a third arm that feeds the
head a cardio embedding shuffled across epochs within the window: the same tensor,
the same statistics, the same capacity, and the same gradient path, with only the
epoch alignment destroyed. If the bypass gain survives shuffling it was never about
the epoch's own breathing; if it falls to the zeroed level, the alignment is what
carried it.

The shuffle is applied inside forward through a wrapper, so the published model
class is untouched and the two original arms run the published code path.

  KMP_DUPLICATE_LIB_OK=TRUE python run_bypass_matched.py
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import mmnet_core as C          # noqa: E402
import sweep                    # noqa: E402

OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final")
FINAL = dict(arm="A+labram", cardio="raw_cnn", cardio_mode="concat",
             temporal="lstm", hidden=256, drop=0.3, lr=3e-4, wd=1e-4)
SEEDS = [42, 1, 7]


def shuffled_forward(self, feeg, fcard):
    """forward() with the bypass block permuted along the epoch axis.

    Copied from MMFeatureNet.forward so the only difference is the permutation;
    the permutation is drawn per batch and applied to every sequence, which keeps
    each window's set of cardio embeddings intact and destroys only which epoch
    each one describes.
    """
    B, Ln = feeg.shape[:2]
    e = self.eeg_enc(feeg.reshape(B * Ln, -1))
    c = self.card_enc(fcard.reshape(B * Ln, -1))
    if self.fusion == "cross":
        fz = self.fuse(e, self.card_proj(c))
    elif self.fusion == "concat":
        fz = self.fuse(torch.cat([e, c], -1))
    else:
        fz = self.fuse(e)
    z = fz.reshape(B, Ln, -1)
    h = self.lstm(z)[0] if self.lstm is not None else self.per_epoch(z)
    c_seq = c.reshape(B, Ln, -1)
    perm = torch.randperm(Ln, device=c_seq.device)
    c_seq = c_seq[:, perm, :]
    return self.stage_head(h), self.apnea_head(torch.cat([h, c_seq], -1)).squeeze(-1)


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "bypass_matched.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time()

    with sweep.config(C, FINAL["arm"], hidden=FINAL["hidden"], drop=FINAL["drop"],
                      lr=FINAL["lr"], wd=FINAL["wd"], cardio=FINAL["cardio"],
                      cardio_mode=FINAL["cardio_mode"]) as h:

        # the capacity claim, checked rather than asserted
        import cardio_cnn
        counts = {}
        for flag in (True, False):
            m = C.MMFeatureNet(n_eeg=h.dim, n_card=h.n_card, hidden=FINAL["hidden"],
                               drop=FINAL["drop"], fusion="concat", bypass=flag,
                               temporal=FINAL["temporal"])
            m.card_enc = cardio_cnn.CardioCNN(d=64, drop=FINAL["drop"])
            counts[flag] = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print("parameters with bypass %s, without %s"
              % (format(counts[True], ","), format(counts[False], ",")))
        assert counts[True] == counts[False], "the two arms are not capacity-matched"
        print("  identical, so the ablation is capacity-matched by construction\n")

        base_forward = C.MMFeatureNet.forward if hasattr(C.MMFeatureNet, "forward") else None
        for arm in ("shuffled",):
            for seed in SEEDS:
                key = "%s|%d" % (arm, seed)
                if key in res:
                    print("[skip] %s" % key, flush=True)
                    continue
                t1 = time.time()
                cls = type(C.MMFeatureNet(n_eeg=h.dim, n_card=h.n_card,
                                          hidden=FINAL["hidden"], drop=FINAL["drop"],
                                          fusion="concat", bypass=True,
                                          temporal=FINAL["temporal"]))
                original = cls.forward
                cls.forward = shuffled_forward
                try:
                    r = C.run_10fold(fusion="concat", temporal=FINAL["temporal"],
                                     seed=seed, bypass=True)
                finally:
                    cls.forward = original
                res[key] = {k: [float(f[k]) for f in r["per_fold"]]
                            for k in ("acc", "kappa", "auc", "ap")}
                res[key]["minutes"] = round((time.time() - t1) / 60, 2)
                json.dump(res, open(path, "w"), indent=1)
                print("%-10s seed %2d  auc %.4f  ap %.4f  acc %.4f  [%.1f min]"
                      % (arm, seed, np.mean(res[key]["auc"]), np.mean(res[key]["ap"]),
                         np.mean(res[key]["acc"]), res[key]["minutes"]), flush=True)

    # ---- compare against the two published arms ----------------------------
    pub = json.load(open(os.path.join(OUT, "bypass_ablation.json")))["per_seed"]

    def fold_means(block, metric):
        rows = [block["seed|%d" % s][metric] for s in SEEDS]
        return np.mean(np.asarray(rows, float), axis=0)

    from scipy.stats import wilcoxon
    print("\n%-10s %9s %9s %9s" % ("metric", "bypass", "shuffled", "zeroed"))
    summary = {}
    for m in ("auc", "ap", "acc", "kappa"):
        with_b = fold_means(pub["with_bypass"], m)
        no_b = fold_means(pub["no_bypass"], m)
        sh = np.mean(np.asarray([res["shuffled|%d" % s][m] for s in SEEDS], float), axis=0)
        p_sh = wilcoxon(with_b, sh).pvalue
        p_zero = wilcoxon(with_b, no_b).pvalue
        summary[m] = dict(with_bypass=round(float(with_b.mean()), 4),
                          shuffled=round(float(sh.mean()), 4),
                          zeroed=round(float(no_b.mean()), 4),
                          p_vs_shuffled=round(float(p_sh), 4),
                          p_vs_zeroed=round(float(p_zero), 4))
        print("%-10s %9.4f %9.4f %9.4f   p(vs shuffled) %.3f  p(vs zeroed) %.3f"
              % (m, with_b.mean(), sh.mean(), no_b.mean(), p_sh, p_zero))

    res["_summary"] = summary
    res["_params"] = {"with_bypass": counts[True], "without_bypass": counts[False]}
    res["_note"] = ("the zeroed arm is bypass=False from bypass_ablation.json; the "
                    "shuffled arm permutes the bypass block across epochs, matching "
                    "capacity, statistics and gradient path while destroying alignment")
    json.dump(res, open(path, "w"), indent=1)
    print("\nsaved -> %s  [%.1f min]" % (path, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
