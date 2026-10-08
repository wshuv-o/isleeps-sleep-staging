"""Lesion-aware MM-Net: the published model plus the lesion input block (lesion.py).

Four configurations, same ten folds and same three seeds as the headline:

    base            the published final model, re-run so per-patient results exist
                    for every seed
    mirror          + ipsilesional channel ordering
    mirror+asym     + hemispheric asymmetry indices
    lesion          + lesion context (side known / bilateral / unknown, NIHSS)

Outputs, under results/revision/runs/lesion/:
    lesion_runs.json     per-fold metrics and per-patient accuracy for each config x seed
    summary.txt          means, paired Wilcoxon on the ten fold-means against base, and
                         accuracy within the lateralised / bilateral / unknown subgroups

    python foundation/run_lesion_aware.py

Needs, under data/ (not in git): mm_features/, multimodal/ and cbramod_emb/labram/,
about 4.2 GB. On a fresh machine, preprocessing/rebuild_data.py downloads the
public release and builds all three. subject_description.xlsx is fetched from
Figshare on first use.
The raw cardio windows take about 3 GB of GPU memory and, on Windows, the same again
in committed system memory, so leave roughly 8 GB of headroom. Results resume:
finished config x seed pairs are skipped on a re-run.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import mmnet_core as C          # noqa: E402
import arms                     # noqa: E402
import sweep                    # noqa: E402
import lesion                   # noqa: E402
from run_final_model import FINAL  # noqa: E402

OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "lesion")
CONFIGS = {
    "base": (),
    "mirror": ("mirror",),
    "mirror+asym": ("mirror", "asym"),
    "lesion": ("mirror", "asym", "context"),
}
METRICS = ("acc", "mf1", "kappa", "auc", "ap")


class ChunkedTensor:
    """A [N, ...] GPU tensor stored as row chunks, indexable like the original.

    train_fold only ever does X[idx] with a 1-D index tensor and reads X.shape[0];
    this returns exactly the rows the single tensor would. It exists because the
    raw cardio windows need one 2.9 GB block, which the 6 GB card cannot always
    supply contiguously even with more than that free.
    """

    def __init__(self, n, tail, dtype, dev, rows=1000):
        import torch
        self.rows, self.shape, self.dtype, self.dev = rows, (n,) + tuple(tail), dtype, dev
        self.chunks = [torch.zeros((min(rows, n - i),) + tuple(tail), dtype=dtype, device=dev)
                       for i in range(0, n, rows)]

    def row(self, i):
        return self.chunks[i // self.rows][i % self.rows]

    def __getitem__(self, idx):
        import torch
        out = torch.empty((len(idx),) + self.shape[1:], dtype=self.dtype, device=self.dev)
        c = torch.div(idx, self.rows, rounding_mode="floor")
        for k in torch.unique(c).tolist():
            m = c == k
            out[m] = self.chunks[k][idx[m] - k * self.rows]
        return out


def lean_windows(subs, stride, eeg_drop, card_drop):
    """mmnet_core.windows, filled straight into GPU tensors one patient at a time.

    Same windows, same order, same zero-padding and mask, so training is unchanged.
    The original stacks every window in host RAM first (2.9 GB with the raw cardio
    stream), which fails on a machine with less free memory than the run that
    produced the published numbers.
    """
    import torch
    L = C.L
    starts = []
    for s in subs:
        n = len(C.DATA[s][2])
        starts.append(list(range(0, max(1, n - L + 1), stride)))
    N = sum(len(x) for x in starts)
    de, dc = C.DATA[subs[0]][0].shape[1], C.DATA[subs[0]][1].shape[1]
    Fe = torch.zeros((N, L, de), dtype=torch.float32, device=C.DEV)
    Fc = ChunkedTensor(N, (L, dc), torch.float32, C.DEV)
    Y = torch.zeros((N, L), dtype=torch.long, device=C.DEV)
    A = torch.zeros((N, L), dtype=torch.float32, device=C.DEV)
    Mk = torch.zeros((N, L), dtype=torch.float32, device=C.DEV)
    i = 0
    for s, st_list in zip(subs, starts):
        fe, fc, y, a = C.DATA[s]
        fe, fc = C.apply_drop(fe, fc, eeg_drop, card_drop)
        for st in st_list:
            k = len(y[st:st + L])
            Fe[i, :k] = torch.from_numpy(fe[st:st + L]).to(C.DEV)
            Fc.row(i)[:k] = torch.from_numpy(fc[st:st + L]).to(C.DEV)
            Y[i, :k] = torch.from_numpy(y[st:st + L]).to(C.DEV)
            A[i, :k] = torch.from_numpy(a[st:st + L].astype(np.float32)).to(C.DEV)
            Mk[i, :k] = 1.0
            i += 1
    return Fe, Fc, Y, A, Mk


_orig_recompile = arms._recompile_with_width


def _recompile_lean(C_, n_eeg, n_card=None):
    out = _orig_recompile(C_, n_eeg, n_card)
    out["windows"] = lean_windows
    return out


arms._recompile_with_width = _recompile_lean


def run_one(steps, seed):
    undo = lesion.install(arms, C, steps) if steps else (lambda: None)
    try:
        with sweep.config(C, FINAL["arm"], hidden=FINAL["hidden"], drop=FINAL["drop"],
                          lr=FINAL["lr"], wd=FINAL["wd"],
                          cardio=FINAL["cardio"], cardio_mode=FINAL["cardio_mode"]) as h:
            r = C.run_10fold(fusion="concat", temporal=FINAL["temporal"], seed=seed)
            dim = h.dim
    finally:
        undo()
    rec = {m: [float(f[m]) for f in r["per_fold"]] for m in METRICS}
    rec["eeg_dim"] = int(dim)
    rec["per_subject"] = {k: v["acc"] for k, v in r["per_subject"].items()}
    return rec


def summarise(res, seeds):
    table = lesion.lesion_table()
    group = {}
    for sid, (side, _) in table.items():
        group["SN%d" % sid] = ("lateralised" if side in ("L", "R")
                               else "bilateral" if side == "B" else "unknown")
    lines = []

    def fold_means(cfg, m):
        return np.mean([res["%s|%d" % (cfg, s)][m] for s in seeds], axis=0)

    lines.append("%-12s %8s %8s %8s %8s %8s   dim" % (("config",) + METRICS))
    for cfg in CONFIGS:
        if not all("%s|%d" % (cfg, s) in res for s in seeds):
            continue
        vals = [np.nanmean(fold_means(cfg, m)) for m in METRICS]
        lines.append("%-12s " % cfg + " ".join("%8.4f" % v for v in vals)
                     + "   %d" % res["%s|%d" % (cfg, seeds[0])]["eeg_dim"])
    lines.append("\npaired Wilcoxon on ten fold-means (seed-averaged), vs base")
    for cfg in CONFIGS:
        if cfg == "base" or not all("%s|%d" % (cfg, s) in res for s in seeds):
            continue
        parts = []
        for m in METRICS:
            d = fold_means(cfg, m) - fold_means("base", m)
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            parts.append("%s %+.4f (p=%.3f, %d/10 up)" % (m, np.nanmean(d), p, int((d > 0).sum())))
        lines.append("%-12s " % cfg + "; ".join(parts))
    lines.append("\nper-patient staging accuracy by lesion group (seed-averaged)")
    for cfg in CONFIGS:
        if not all("%s|%d" % (cfg, s) in res for s in seeds):
            continue
        ps = {}
        for s in seeds:
            for k, v in res["%s|%d" % (cfg, s)]["per_subject"].items():
                ps.setdefault(k, []).append(v)
        row = []
        for g in ("lateralised", "bilateral", "unknown"):
            v = [np.mean(x) for k, x in ps.items() if group.get(k) == g]
            row.append("%s %.4f (n=%d)" % (g, np.mean(v), len(v)))
        lines.append("%-12s " % cfg + "; ".join(row))
    # paired per-patient test, lateralised patients, full lesion model vs base
    for cfg in ("mirror", "mirror+asym", "lesion"):
        if not all("%s|%d" % (c, s) in res for c in (cfg, "base") for s in seeds):
            continue
        keys = [k for k in res["base|%d" % seeds[0]]["per_subject"] if group.get(k) == "lateralised"]
        a = np.array([np.mean([res["%s|%d" % (cfg, s)]["per_subject"][k] for s in seeds]) for k in keys])
        b = np.array([np.mean([res["base|%d" % s]["per_subject"][k] for s in seeds]) for k in keys])
        lines.append("lateralised, %s vs base: %+.4f, Wilcoxon p=%.4f, %d/%d patients up"
                     % (cfg, np.mean(a - b), wilcoxon(a - b).pvalue, int((a > b).sum()), len(keys)))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "lesion_runs.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for cfg in a.configs:
        for seed in a.seeds:
            key = "%s|%d" % (cfg, seed)
            if key in res:
                print("[skip] %s" % key, flush=True)
                continue
            t0 = time.time()
            res[key] = run_one(CONFIGS[cfg], seed)
            res[key]["minutes"] = (time.time() - t0) / 60
            json.dump(res, open(path, "w"), indent=1)
            print("[ok] %-12s seed %d  acc %.4f  kappa %.4f  auc %.4f  dim %d  [%.1f min]"
                  % (cfg, seed, np.mean(res[key]["acc"]), np.mean(res[key]["kappa"]),
                     np.nanmean(res[key]["auc"]), res[key]["eeg_dim"], res[key]["minutes"]),
                  flush=True)
    text = summarise(res, a.seeds)
    open(os.path.join(OUT, "summary.txt"), "w").write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
