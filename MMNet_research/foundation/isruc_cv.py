"""MM-Net trained and tested on ISRUC-Sleep, subject-independent cross-validation.

Recordings: external_final.py build output (EXT/isruc/<grp>/<tag>.npz), groups given by --groups.
Folds: subjects (both nights of a subgroup II subject together) shuffled with RandomState(42) and
dealt into --folds folds; in each fold two training subjects (RandomState(100+k)) are validation.

  finetune   LaBraM with E1/E2 as F7/F8 fine-tuned on the fold's training recordings (early stopping on
             validation) -> EXT/isruc_cv/ft/fold<k>; --part p: cross-fitted part p of 3 (split by subject)
             -> EXT/isruc_cv/cf/fold<k>
  align      ridge map (lambda 0.01) of each cross-fitted part into the full encoder's space
             -> EXT/isruc_cv/fta/fold<k>
  mmnet      --config final: no class weights, cross fusion, aligned fine-tuned encoder, 8 channels
             --config frozen: published configuration (class weights, concat, frozen LaBraM, 7 channels)
             seeds 42/1/7, HMM fitted on the training recordings. AHI: expected event onsets per hour of
             predicted sleep x k, k = median ratio on the fold's validation recordings.

    G:/codes/isleeps/prepenv/Scripts/python isruc_cv.py finetune --fold k [--part p]
    python isruc_cv.py align | mmnet --config final|frozen
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

EXT = "H:/isleeps_ext"
CV = os.path.join(EXT, "isruc_cv")
RES = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "lesion", "external")
S = (42, 1, 7)


def recs(groups):
    return {os.path.basename(f)[:-4]: f for g in groups for f in sorted(glob.glob(os.path.join(EXT, "isruc", g, "*.npz")))}


def subject(tag):
    return "_".join(tag.split("_")[:2])


def folds(groups, n):
    R = sorted(recs(groups))
    subs = sorted({subject(t) for t in R})
    rng = np.random.RandomState(42)
    order = list(subs); rng.shuffle(order)
    out = []
    for k in range(n):
        te_s = order[k::n]
        tr_s = [s for s in order if s not in te_s]
        rs = np.random.RandomState(100 + k); tr_s = list(tr_s); rs.shuffle(tr_s)
        va_s = tr_s[:2]
        out.append(([t for t in R if subject(t) in tr_s[2:]], [t for t in R if subject(t) in va_s],
                    [t for t in R if subject(t) in te_s]))
    return out


def cf_parts(k, tr):
    subs = sorted({subject(t) for t in tr})
    rng = np.random.RandomState(1000 + k); rng.shuffle(subs)
    p = {s: i % 3 for i, s in enumerate(subs)}
    return {t: p[subject(t)] for t in tr}


# --------------------------------------------------------------------------- encoder
def finetune(k, groups, n, part=None, passes=3.0, bs=64, lr=3e-5, seed=42):
    import torch
    import torch.nn as nn
    import finetune_labram as FL
    FL.CH, FL.NCH = FL.CH + ["F7", "F8"], 6
    torch.manual_seed(seed); np.random.seed(seed)
    R = recs(groups)
    tr, va, te = folds(groups, n)[k]
    held = []
    if part is not None:
        p = cf_parts(k, tr)
        held = [t for t in tr if p[t] == part]
        tr = [t for t in tr if p[t] != part]
    X, Y = {}, {}
    for t, f in R.items():
        d = np.load(f)
        X[t] = np.ascontiguousarray(d["x"][:, :6]).astype(np.float16); Y[t] = d["y"].astype(np.int64)
    idx = np.array([(t, i) for t in tr for i in range(len(Y[t]))], dtype=object)
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
        x = torch.from_numpy(np.stack([X[t][int(i)] for t, i in b])).to(FL.DEV)
        y = torch.tensor([Y[t][int(i)] for t, i in b], device=FL.DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = ce(model(x).float(), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        if (step + 1) % every == 0 or step == steps - 1:
            P, _ = FL.predict(model, X, va)
            acc = float(np.mean(np.concatenate([P[t].argmax(1) == Y[t] for t in va])))
            if acc > best:
                best, best_state = acc, {kk: v.detach().clone() for kk, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    if part is not None:
        P, E = FL.predict(model, X, held); od = os.path.join(CV, "cf", "fold%d" % k)
    else:
        P, E = FL.predict(model, X, list(R)); od = os.path.join(CV, "ft", "fold%d" % k)
    os.makedirs(od, exist_ok=True)
    for t in P:
        np.savez_compressed(os.path.join(od, t + ".npz"), emb=E[t].astype(np.float16), prob=P[t].astype(np.float32))
    msg = ""
    if part is None:
        msg = ", test (one epoch, no HMM) %.4f" % float(np.mean(np.concatenate([P[t].argmax(1) == Y[t] for t in te])))
    print("fold %d %s: best val %.4f%s" % (k, "part %d" % part if part is not None else "full", best, msg), flush=True)


def align(groups, n, lam=0.01):
    R = recs(groups)
    for k, (tr, va, te) in enumerate(folds(groups, n)):
        part = cf_parts(k, tr)
        ld = lambda sub, t: np.load(os.path.join(CV, sub, "fold%d" % k, t + ".npz"))["emb"].astype(np.float64)
        maps = {}
        for p in range(3):
            rs = [t for t in tr if part[t] == p]
            Xc = np.concatenate([ld("cf", t) for t in rs]); Yc = np.concatenate([ld("ft", t) for t in rs])
            mx, my = Xc.mean(0), Yc.mean(0)
            maps[p] = (mx, my, np.linalg.solve((Xc - mx).T @ (Xc - mx) + lam * len(Xc) * np.eye(Xc.shape[1]), (Xc - mx).T @ (Yc - my)))
        od = os.path.join(CV, "fta", "fold%d" % k); os.makedirs(od, exist_ok=True)
        for t in R:
            if t in part:
                mx, my, W = maps[part[t]]; e = (ld("cf", t) - mx) @ W + my
            else:
                e = ld("ft", t)
            np.savez_compressed(os.path.join(od, t + ".npz"), emb=np.asarray(e, np.float16))
        print("fold %d aligned" % k, flush=True)


# --------------------------------------------------------------------------- MM-Net
def mmnet(config, groups, n):
    import torch
    import run_lesion_aware as RL
    import cardio_cnn
    from run_event_ahi import metrics as ahi_metrics
    from sklearn.metrics import accuracy_score, average_precision_score, cohen_kappa_score, f1_score, roc_auc_score
    C, sweep, FINAL = RL.C, RL.sweep, RL.FINAL
    final = config == "final"
    n_ch = 8 if final else 7
    R = recs(groups)
    raw = {t: np.load(f) for t, f in R.items()}
    orig_cw, orig_table, orig_nch = C.sqrt_cw, cardio_cnn.raw_cardio_table, cardio_cnn.N_CH
    if final:
        import run_mmnet_pflow as P
        C.sqrt_cw = lambda tr: torch.ones(5, dtype=torch.float32, device=C.DEV)
        cardio_cnn.raw_cardio_table, cardio_cnn.N_CH = P.table_with_pressure, 8

    def table(k):
        out = {}
        for t, d in raw.items():
            fe = d["Feeg"].astype(np.float32); fe = (fe - fe.mean(0)) / (fe.std(0) + 1e-6)
            ep = os.path.join(CV, "fta", "fold%d" % k, t + ".npz") if final else os.path.join(EXT, "emb", "labram", t + ".npz")
            e = np.load(ep)["emb"].astype(np.float32); e = (e - e.mean(0)) / (e.std(0) + 1e-6)
            c = d["card"][:, :n_ch].astype(np.float32)
            c = (c - c.mean(axis=(0, 2), keepdims=True)) / (c.std(axis=(0, 2), keepdims=True) + 1e-6)
            out[t] = (np.concatenate([fe, e], 1), c.reshape(len(c), -1), d["y"].astype(np.int64), d["apnea"].astype(np.int64))
        return out

    def ahi_true(t):
        y, nev = raw[t]["y"], raw[t]["n_events"]
        return float(nev[y != 0].sum() / max((y != 0).sum() * 30 / 3600, 1e-6))

    probs, aprobs, stages, ahi_pred = {}, {}, {}, {}
    saved = dict(C.DATA)
    try:
        with sweep.config(C, FINAL["arm"], hidden=FINAL["hidden"], drop=FINAL["drop"], lr=FINAL["lr"], wd=FINAL["wd"],
                          cardio=FINAL["cardio"], cardio_mode=FINAL["cardio_mode"]):
            for k, (tr, va, te) in enumerate(folds(groups, n)):
                t0 = time.time()
                C.DATA.clear(); C.DATA.update(table(k))
                Am = np.ones((C.NC, C.NC)); pi = np.ones(C.NC)
                for t in tr:
                    y = C.DATA[t][2]; pi[y[0]] += 1
                    for a_, b_ in zip(y[:-1], y[1:]): Am[a_, b_] += 1
                A_log, pi_log = np.log(Am / Am.sum(1, keepdims=True)), np.log(pi / pi.sum())
                pk = {t: [] for t in va + te}; ak = {t: [] for t in va + te}
                for sd in S:
                    model = C.train_fold(tr, va, "cross" if final else "concat", [], [], seed=sd, temporal=FINAL["temporal"])
                    for t in va + te:
                        sp, ap = C.subj_infer(model, t, [], [])
                        pk[t].append(sp); ak[t].append(ap)
                    del model; torch.cuda.empty_cache()
                est = {}
                for t in va + te:
                    st = C.hmm(A_log, pi_log, np.log(np.mean(pk[t], 0) + C.EPS)); apm = np.mean(ak[t], 0)
                    est[t] = apm[st != 0].sum() / max((st != 0).sum() * 30 / 3600, 0.5)
                    if t in te:
                        stages[t] = st; probs[t] = [p.astype(np.float32) for p in pk[t]]; aprobs[t] = apm
                        for i, sd in enumerate(S):
                            stages["%s|%d" % (t, sd)] = C.hmm(A_log, pi_log, np.log(pk[t][i] + C.EPS))
                kk = float(np.median([ahi_true(t) / max(est[t], 1e-3) for t in va]))
                for t in te:
                    ahi_pred[t] = kk * est[t]
                acc = np.mean(np.concatenate([stages[t] == raw[t]["y"] for t in te]))
                print("%s fold %d: %d test recordings, 3-seed acc %.4f, AHI k %.3f [%.1f min]"
                      % (config, k, len(te), acc, kk, (time.time() - t0) / 60), flush=True)
    finally:
        C.DATA.clear(); C.DATA.update(saved)
        C.sqrt_cw, cardio_cnn.raw_cardio_table, cardio_cnn.N_CH = orig_cw, orig_table, orig_nch
    out = {"config": config, "groups": groups, "folds": n}
    lines = ["ISRUC %s, MM-Net %s trained and tested on ISRUC (%d subject-independent folds)" % ("+".join(groups), config, n)]
    for g in groups + (["+".join(groups)] if len(groups) > 1 else []):
        ts = [t for t in R if t.split("_")[0] in g.split("+")]
        yt = np.concatenate([raw[t]["y"] for t in ts])
        m = lambda yp: dict(acc=float(accuracy_score(yt, yp)), mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                            kappa=float(cohen_kappa_score(yt, yp)))
        ens = m(np.concatenate([stages[t] for t in ts]))
        single = float(np.mean([m(np.concatenate([stages["%s|%d" % (t, sd)] for t in ts]))["acc"] for sd in S]))
        at = np.concatenate([raw[t]["apnea"] for t in ts]); ap = np.concatenate([aprobs[t] for t in ts])
        auc = float(roc_auc_score(at, ap)) if len(np.unique(at)) > 1 else None
        apv = float(average_precision_score(at, ap)) if auc else None
        am = ahi_metrics([ahi_pred[t] for t in ts], [ahi_true(t) for t in ts])
        out[g] = dict(n=len(ts), single_acc=single, ensemble=ens, resp_auc=auc, resp_ap=apv, ahi=am,
                      ahi_true={t: ahi_true(t) for t in ts}, ahi_pred={t: float(ahi_pred[t]) for t in ts})
        lines.append("%-8s %2d rec  staging single %.4f | 3-seed acc %.4f mf1 %.4f kappa %.4f | resp AUC %s AP %s | AHI rho %.3f MAE %.1f sev acc %.2f kappa %.2f"
                     % (g, len(ts), single, ens["acc"], ens["mf1"], ens["kappa"], "%.3f" % auc if auc else "n/a",
                        "%.3f" % apv if apv else "n/a", am["rho"], am["mae"], am["sev_acc"], am["sev_kappa"]))
    os.makedirs(RES, exist_ok=True)
    name = "isruc_cv_%s_%s" % ("".join(groups), config)
    json.dump(out, open(os.path.join(RES, name + ".json"), "w"), indent=1)
    open(os.path.join(RES, name + ".txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["finetune", "align", "mmnet"])
    ap.add_argument("--groups", nargs="+", default=["II", "III"])
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--part", type=int, default=None)
    ap.add_argument("--config", default="final", choices=["final", "frozen"])
    a = ap.parse_args()
    if a.stage == "finetune":
        finetune(a.fold, a.groups, a.folds, a.part)
    elif a.stage == "align":
        align(a.groups, a.folds)
    else:
        mmnet(a.config, a.groups, a.folds)
