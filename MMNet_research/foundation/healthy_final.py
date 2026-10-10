"""The final architecture's encoder on the healthy cohort (Sleep-EDF, 40 recordings / 21 subjects).

Same steps as on iSLEEPS, on run_healthy_cv.py's subject-independent folds:
  finetune   LaBraM fine-tuned per fold on the training recordings (finetune_labram.Stager), with
             the paper's Sleep-EDF mapping: Fpz-Cz, Pz-Oz, Fpz-Cz, Pz-Oz as C4 C3 O2 O1 and the
             horizontal EOG as F7 and F8 (the final architecture's EEG + EOG encoder)
  crossfit   three parts per fold, split by SUBJECT (both nights of a person on the same side);
             each training recording embedded by an encoder that never saw that subject
  align      ridge map of each cross-fitted encoder into the full-fold encoder's space
  mmnet      MM-Net neural stream (no cardiorespiratory channels on Sleep-EDF), no class weights,
             cross fusion, seeds 42/1/7: frozen LaBraM vs the fine-tuned, aligned encoder

    G:/codes/isleeps/prepenv/Scripts/python healthy_final.py finetune|crossfit   (braindecode)
    python healthy_final.py align|mmnet
"""
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

DATA = os.path.join(REPO, "data")
PROC = os.path.join(DATA, "sleep_edf_proc")
RES = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "lesion")
N_FOLDS = 10
IDX = [0, 1, 0, 1, 2, 2]
CH = ["C4", "C3", "O2", "O1", "F7", "F8"]


def subject_of(rec):
    return rec[3:5]


def folds():
    recs = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(PROC, "*.npz")))
    subs = sorted({subject_of(r) for r in recs})
    rng = np.random.RandomState(42)
    order = list(subs); rng.shuffle(order)
    out = []
    for fi, te_subs in enumerate([order[i::N_FOLDS] for i in range(N_FOLDS)]):
        te = [r for r in recs if subject_of(r) in te_subs]
        tr_all = [r for r in recs if subject_of(r) not in te_subs]
        rs = np.random.RandomState(100 + fi)
        tr_all = list(tr_all); rs.shuffle(tr_all)
        nv = max(2, len(tr_all) // 9)
        out.append((tr_all[nv:], tr_all[:nv], te))
    return recs, out


def cf_parts(k, tr):
    subs = sorted({subject_of(r) for r in tr})
    rng = np.random.RandomState(1000 + k)
    rng.shuffle(subs)
    part = {s: i % 3 for i, s in enumerate(subs)}
    return {r: part[subject_of(r)] for r in tr}


# --------------------------------------------------------------------------- fine-tuning
def finetune(k, cf_part=None, passes=3.0, bs=64, lr=3e-5, seed=42):
    import torch
    import torch.nn as nn
    import finetune_labram as FL
    FL.CH, FL.NCH = CH, len(CH)
    torch.manual_seed(seed); np.random.seed(seed)
    recs, F = folds()
    tr, va, te = F[k]
    held = []
    if cf_part is not None:
        part = cf_parts(k, tr)
        held = [r for r in tr if part[r] == cf_part]
        tr = [r for r in tr if part[r] != cf_part]
    X, Y = {}, {}
    for r in recs:
        d = np.load(os.path.join(PROC, r + ".npz"))
        X[r] = np.ascontiguousarray(d["x"][:, IDX], dtype=np.float16); Y[r] = d["y"].astype(np.int64)   # Sleep-EDF arrays are column-major
    idx = np.array([(r, i) for r in tr for i in range(len(Y[r]))], dtype=object)
    model = FL.Stager().to(FL.DEV)
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": lr},
                             {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.05)
    steps = int(passes * len(idx) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, 1e-3], total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(); ce = nn.CrossEntropyLoss()
    every = max(1, int(len(idx) / bs / 4)); best, best_state = -1.0, None
    for step in range(steps):
        model.train()
        b = idx[np.random.randint(0, len(idx), bs)]
        x = torch.from_numpy(np.stack([X[r][int(i)] for r, i in b])).to(FL.DEV)
        y = torch.tensor([Y[r][int(i)] for r, i in b], device=FL.DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = ce(model(x).float(), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        if (step + 1) % every == 0 or step == steps - 1:
            P, _ = FL.predict(model, X, va)
            acc = float(np.mean(np.concatenate([P[r].argmax(1) == Y[r] for r in va])))
            if acc > best:
                best, best_state = acc, {kk: v.detach().clone() for kk, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    if cf_part is not None:
        P, E = FL.predict(model, X, held)
        od = os.path.join(DATA, "labram_cf_sedf", "fold%d" % k)
    else:
        P, E = FL.predict(model, X, recs)
        od = os.path.join(DATA, "labram_ft_sedf", "fold%d" % k)
    os.makedirs(od, exist_ok=True)
    for r in P:
        np.savez_compressed(os.path.join(od, r + ".npz"), emb=E[r].astype(np.float16),
                            prob=P[r].astype(np.float32), y=Y[r])
    tea = None
    if cf_part is None:
        tea = float(np.mean(np.concatenate([P[r].argmax(1) == Y[r] for r in te])))
    print("fold %d %s: best val %.4f%s" % (k, "part %d" % cf_part if cf_part is not None else "full",
                                          best, "" if tea is None else ", test (one epoch, no HMM) %.4f" % tea), flush=True)


# --------------------------------------------------------------------------- alignment
def align(k, lam=0.01):
    recs, F = folds()
    tr, va, te = F[k]
    part = cf_parts(k, tr)
    maps = {}
    for p in range(3):
        rs = [r for r in tr if part[r] == p]
        Xc = np.concatenate([np.load(os.path.join(DATA, "labram_cf_sedf", "fold%d" % k, r + ".npz"))["emb"] for r in rs]).astype(np.float64)
        Yc = np.concatenate([np.load(os.path.join(DATA, "labram_ft_sedf", "fold%d" % k, r + ".npz"))["emb"] for r in rs]).astype(np.float64)
        mx, my = Xc.mean(0), Yc.mean(0)
        W = np.linalg.solve((Xc - mx).T @ (Xc - mx) + lam * len(Xc) * np.eye(Xc.shape[1]), (Xc - mx).T @ (Yc - my))
        maps[p] = (mx, my, W)
    od = os.path.join(DATA, "labram_ext", "sedf_fta_f%d" % k)
    os.makedirs(od, exist_ok=True)
    for r in recs:
        if r in part:
            e = np.load(os.path.join(DATA, "labram_cf_sedf", "fold%d" % k, r + ".npz"))["emb"].astype(np.float64)
            mx, my, W = maps[part[r]]
            e = (e - mx) @ W + my
        else:
            e = np.load(os.path.join(DATA, "labram_ft_sedf", "fold%d" % k, r + ".npz"))["emb"]
        np.savez_compressed(os.path.join(od, r + ".npz"), emb=np.asarray(e, np.float16))
    print("fold %d aligned -> %s" % (k, od), flush=True)


# --------------------------------------------------------------------------- MM-Net
def mmnet(variant, seeds=(42, 1, 7)):
    import torch
    import mmnet_core as C
    import sweep
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    import run_healthy_cv as H
    recs, F = folds()
    orig_cw = C.sqrt_cw
    C.sqrt_cw = lambda tr: torch.ones(5, dtype=torch.float32, device=C.DEV)      # no class weights
    rows = {sd: [] for sd in seeds}; ens = []
    saved = dict(C.DATA)
    try:
        with sweep.config(C, H.FINAL["arm"], hidden=H.FINAL["hidden"], drop=H.FINAL["drop"],
                          lr=H.FINAL["lr"], wd=H.FINAL["wd"], cardio=H.FINAL["cardio"], cardio_mode="concat") as h:
            for fi, (tr, va, te) in enumerate(F):
                ldir = os.path.join(DATA, "labram_ext", "sleep_edf" if variant == "frozen" else "sedf_fta_f%d" % fi)
                data = H.load(h.n_card, ldir)
                C.DATA.clear(); C.DATA.update(data)
                Am = np.ones((C.NC, C.NC)); pi = np.ones(C.NC)
                for s in tr:
                    y = C.DATA[s][2]; pi[y[0]] += 1
                    for x, z in zip(y[:-1], y[1:]): Am[x, z] += 1
                A_log, pi_log = np.log(Am / Am.sum(1, keepdims=True)), np.log(pi / pi.sum())
                yt = np.concatenate([C.DATA[s][2] for s in te]); probs = {s: [] for s in te}
                for sd in seeds:
                    model = C.train_fold(tr, va, "cross", [], [], seed=sd, temporal="lstm", task="stage")
                    yp = []
                    for s in te:
                        sp, _ = C.subj_infer(model, s, [], [])
                        probs[s].append(sp); yp.append(C.hmm(A_log, pi_log, np.log(sp + C.EPS)))
                    rows[sd].append(accuracy_score(yt, np.concatenate(yp)))
                    del model; torch.cuda.empty_cache()
                yp = np.concatenate([C.hmm(A_log, pi_log, np.log(np.mean(probs[s], 0) + C.EPS)) for s in te])
                ens.append(dict(acc=accuracy_score(yt, yp), mf1=f1_score(yt, yp, average="macro", zero_division=0),
                                kappa=cohen_kappa_score(yt, yp)))
                print("%s fold %d: single %.4f | ensemble %.4f" % (variant, fi, np.mean([rows[sd][-1] for sd in seeds]), ens[-1]["acc"]), flush=True)
    finally:
        C.DATA.clear(); C.DATA.update(saved); C.sqrt_cw = orig_cw
    out = dict(variant=variant, single=[float(np.mean([rows[sd][fi] for sd in seeds])) for fi in range(N_FOLDS)],
               ensemble={m: [float(e[m]) for e in ens] for m in ("acc", "mf1", "kappa")})
    json.dump(out, open(os.path.join(RES, "healthy_final_%s.json" % variant), "w"), indent=1)
    print("%s: single-model %.4f | 3-seed acc %.4f mf1 %.4f kappa %.4f" % (variant, np.mean(out["single"]),
          *(np.mean(out["ensemble"][m]) for m in ("acc", "mf1", "kappa"))), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["finetune", "crossfit", "align", "mmnet"])
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--part", type=int, default=None)
    ap.add_argument("--variant", default="aligned", choices=["frozen", "aligned"])
    a = ap.parse_args()
    if a.stage == "finetune":
        finetune(a.fold)
    elif a.stage == "crossfit":
        finetune(a.fold, cf_part=a.part)
    elif a.stage == "align":
        for k in ([a.fold] if a.fold is not None else range(N_FOLDS)):
            align(k)
    else:
        mmnet(a.variant)
