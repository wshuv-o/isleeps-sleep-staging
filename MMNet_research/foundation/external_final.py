"""The final architecture on ISRUC-Sleep, trained on all 99 iSLEEPS patients, applied unchanged.

Stages (outputs under EXT = H:/isleeps_ext unless --ext is given):

  build    ISRUC subgroup (I, II, III) -> EXT/isruc/<grp>/<tag>.npz
             x      [n, 7, 3000] uV @ 100 Hz: C4-M1 C3-M2 O2-M1 O1-M2, LOC, ROC, chin (processed7 order)
             Feeg   [n, 188] extract_features_v2, as iSLEEPS mm_features
             (EEG/EOG/chin x10 where the header declares +-25 uV; see build())
             card   [n, 8, 750] @ 25 Hz: ECG, Flow (thermistor), Thorax, Abdomen, Effort (= Thorax +
                    Abdomen, iSLEEPS "Sum Effort"), SpO2, Pulse (heart rate from ECG R-peaks; ISRUC has
                    no oximeter pulse), nasal pressure
             y, apnea (1 if an apnea/hypopnea is scored in the epoch), n_events (count per epoch)
           Epochs with a stage outside W/N1/N2/N3/R are dropped after the signal is cut on the full
           grid, as processed7 does for iSLEEPS. Annotations: rater 1 (<rec>_1.xlsx).
  labram   (prep venv) frozen LaBraM embedding of ISRUC (build_labram_cache.embed) -> EXT/emb/labram;
           LaBraM fine-tuned with E1/E2 as F7/F8 on all 99 patients (validation: 11 patients), plus
           three cross-fitted parts -> EXT/labram_ft_eog_all, EXT/labram_cf_eog_all, and ISRUC
           embedded by the full encoder -> EXT/emb/fta_eog
  align    ridge map (lambda 0.01) of each cross-fitted part into the full encoder's space ->
           data/cbramod_emb/labramfta_eog_all (training input of the final MM-Net)
  mmnet    MM-Net trained on all 99 (same 11 validation patients), seeds 42/1/7, HMM fitted on the
           training patients; ISRUC scored with no adaptation. Configurations:
             final   base-nocw-cross-ftlae-p8-al  (fine-tuned aligned encoder, 8 channels)
             frozen  published MM-Net: class weights, concat fusion, frozen LaBraM, 7 channels
           AHI: MM-Net expected event onsets per hour of predicted sleep x k, where k is the median
           over the ten iSLEEPS folds of the validation-patient scale factor (ahi_combo.py's k);
           nothing is fitted on ISRUC. Reference AHI = scored apneas + hypopneas in scored sleep
           epochs per hour of scored sleep.

    python external_final.py build --group III
    G:/codes/isleeps/prepenv/Scripts/python external_final.py labram [--part 0|1|2|full|frozen]
    python external_final.py align
    MMNET_FEATURES=aligned/mm_features MMNET_MULTIMODAL=aligned/multimodal MMNET_PFLOW=aligned/pressure_flow \
        python external_final.py mmnet --config final|frozen --groups II III
"""
import argparse
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "preprocessing"))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "utils"))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

DATA = os.path.join(REPO, "data")
RES = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "lesion", "external")
RAW = "H:/data"
EXT = "H:/isleeps_ext"
STAGE = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
RESP = re.compile(r"\b(OA|CA|MA|OH|CH|MH)\b")
S = (42, 1, 7)


# =========================================================================== build
def recordings(grp):
    root = os.path.join(RAW, "ISRUC-Sleep-%s" % grp, "ISRUC-Sleep-%s" % grp)
    out = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*.rec"), recursive=True)):
        parts = os.path.relpath(p, root).replace("\\", "/").split("/")
        tag = "%s_S%s" % (grp, parts[0]) + ("_%s" % parts[1] if len(parts) == 3 else "")
        xlsx = p[:-4] + "_1.xlsx"
        out.append((tag, p, xlsx))
    return out


def read_annotations(xlsx):
    import openpyxl
    ws = openpyxl.load_workbook(xlsx, data_only=True, read_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    if rows and not isinstance(rows[0][0], (int, float)):          # header row (subgroup I/II)
        rows = rows[1:]
    stage, nev = [], []
    for r in rows:
        if r[0] is None:
            continue
        stage.append(STAGE.get(str(r[1]).strip(), -1) if r[1] is not None else -1)
        nev.append(len(RESP.findall(str(r[4] or ""))))
    return np.array(stage, np.int64), np.array(nev, np.int64)


def declared_max(path, i):
    """physical maximum the EDF header declares for signal i."""
    with open(path, "rb") as f:
        h = f.read(256); ns = int(h[252:256])
        f.seek(256 + ns * (16 + 80 + 8 + 8) + 8 * i)
        return float(f.read(8))


def heart_rate(ecg, fs, n_out, fs_out=25):
    """Beat-to-beat heart rate (bpm) from ECG R-peaks, as a continuous trace at fs_out."""
    from scipy.signal import butter, find_peaks, medfilt, sosfiltfilt
    x = sosfiltfilt(butter(3, [5, 25], btype="band", fs=fs, output="sos"), ecg)
    e = np.gradient(x) ** 2
    w = max(1, int(0.12 * fs))
    e = np.convolve(e, np.ones(w) / w, mode="same")
    thr = 0.3 * np.percentile(e, 99)
    pk, _ = find_peaks(e, height=thr, distance=int(0.3 * fs))
    t_out = np.arange(n_out) / fs_out
    if len(pk) < 10:
        return np.zeros(n_out, np.float32)
    rr = np.diff(pk) / fs
    hr = 60.0 / rr
    ok = (hr > 30) & (hr < 200)
    t = pk[1:][ok] / fs
    hr = medfilt(hr[ok], 5)
    return np.interp(t_out, t, hr).astype(np.float32)


def build(grp):
    import build_isruc_external as B
    from features_v2 import extract_features_v2
    od = os.path.join(EXT, "isruc", grp)
    os.makedirs(od, exist_ok=True)
    for tag, rec, xlsx in recordings(grp):
        dst = os.path.join(od, tag + ".npz")
        if os.path.exists(dst):
            print("[skip]", tag); continue
        lab, trd, rates, sigs = B.read_edf(rec)
        idx = B.pick_channels(lab, trd)
        fl2 = [i for i, t in enumerate(trd) if t.upper().startswith("FLOW2")]
        missing = [k for k in ("c0", "c1", "c2", "c3", "c4", "c5", "c6") if k not in idx]
        if missing:
            print("[FAIL] %s: missing %s (labels %s)" % (tag, missing, lab)); continue
        stage, nev = read_annotations(xlsx)
        n_sig = int(len(sigs[idx["c0"]]) / rates[idx["c0"]] / 30)
        n = min(len(stage), n_sig)
        stage, nev = stage[:n], nev[:n]
        neural = np.stack([B.epoch(B.to_rate(sigs[idx["c%d" % k]], rates[idx["c%d" % k]], 100), 100, n)
                           for k in range(7)], 1).astype(np.float32)
        # Some ISRUC files (all of subgroup III, two of II) declare EEG/EOG at +-25 uV; their N3
        # amplitude is ~10x below the other files' and below the 75 uV that defines N3 -> +-250 uV
        gain = 10.0 if declared_max(rec, idx["c0"]) <= 30 else 1.0
        neural *= gain
        card = np.zeros((n, 8, 750), np.float32)
        valid = np.zeros(8, bool)

        def put(j, i):
            card[:, j] = B.epoch(B.to_rate(sigs[i], rates[i], 25), 25, n); valid[j] = True
        for j, name in enumerate(("ECG", "Flow", "Thorax", "Abdomen")):
            if name in idx:
                put(j, idx[name])
        if valid[2] and valid[3]:
            card[:, 4] = card[:, 2] + card[:, 3]; valid[4] = True
        if "SpO2" in idx:
            put(5, idx["SpO2"])
        if "ECG" in idx:
            i = idx["ECG"]
            card[:, 6] = heart_rate(sigs[i], rates[i], n * 750).reshape(n, 750); valid[6] = bool(card[:, 6].any())
        if fl2:
            put(7, fl2[0])
        keep = stage >= 0
        neural, card, y, nev = neural[keep], card[keep], stage[keep], nev[keep]
        Feeg, _ = extract_features_v2(neural)
        np.savez_compressed(dst, x=neural.astype(np.float16), Feeg=np.nan_to_num(Feeg).astype(np.float32),
                            card=card, cvalid=valid, y=y, apnea=(nev > 0).astype(np.int64),
                            n_events=nev, dropped=int((~keep).sum()), gain=gain)
        sl = y != 0
        ahi = nev[sl].sum() / max(sl.sum() * 30 / 3600, 1e-6)
        print("[ok] %-12s %4d epochs (%d dropped)  events %3d  AHI %5.1f  gain %g  channels %s"
              % (tag, len(y), (~keep).sum(), nev.sum(), ahi, gain, "".join("1" if v else "0" for v in valid)), flush=True)


# =========================================================================== LaBraM
def all_split():
    import mmnet_core as C
    rng = np.random.RandomState(110)
    subs = list(C.SUBS)
    rng.shuffle(subs)
    nv = max(10, len(subs) // 9)
    return subs[nv:], subs[:nv]


def cf_part_of(tr):
    rng = np.random.RandomState(1099)
    order = sorted(tr)
    rng.shuffle(order)
    return {s: i % 3 for i, s in enumerate(order)}


def ext_recs(groups):
    return {os.path.basename(f)[:-4]: f for g in groups for f in sorted(glob.glob(os.path.join(EXT, "isruc", g, "*.npz")))}


def labram(part, groups, passes=3.0, bs=64, lr=3e-5, seed=42):
    import torch
    import torch.nn as nn
    import finetune_labram as FL
    if part == "frozen":
        import build_labram_cache as BLC
        m = BLC.encoder(pretrained=True)
        od = os.path.join(EXT, "emb", "labram"); os.makedirs(od, exist_ok=True)
        for tag, f in ext_recs(groups).items():
            np.savez_compressed(os.path.join(od, tag + ".npz"), emb=BLC.embed(m, np.load(f)["x"]).astype(np.float16))
        print("frozen embeddings ->", od, flush=True)
        return
    FL.CH, FL.NCH = FL.CH + ["F7", "F8"], 6
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = all_split()
    if part != "full":
        p = cf_part_of(tr)
        held = [s for s in tr if p[s] == int(part)]
        tr = [s for s in tr if p[s] != int(part)]
    X, Y = FL.load_eeg()
    idx = np.array([(s, i) for s in tr for i in range(len(Y[s]))])
    print("LaBraM-EOG, all patients, part %s: %d train patients (%d epochs), %d validation"
          % (part, len(tr), len(idx), len(va)), flush=True)
    model = FL.Stager().to(FL.DEV)
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": lr},
                             {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.05)
    steps = int(passes * len(idx) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, 1e-3], total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(); ce = nn.CrossEntropyLoss()
    every = max(1, int(len(idx) / bs / 4)); best, best_state = -1.0, None
    t0 = time.time()
    for step in range(steps):
        model.train()
        b = idx[np.random.randint(0, len(idx), bs)]
        x = torch.from_numpy(np.stack([X[int(s)][int(i)] for s, i in b])).to(FL.DEV)
        y = torch.tensor([Y[int(s)][int(i)] for s, i in b], device=FL.DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = ce(model(x).float(), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        if (step + 1) % every == 0 or step == steps - 1:
            P, _ = FL.predict(model, X, va)
            acc = FL.metrics(Y, P, va)["acc"]
            print("  step %5d/%d  val acc %.4f  [%.1f min]" % (step + 1, steps, acc, (time.time() - t0) / 60), flush=True)
            if acc > best:
                best, best_state = acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    if part != "full":
        P, E = FL.predict(model, X, held)
        od = os.path.join(EXT, "labram_cf_eog_all"); os.makedirs(od, exist_ok=True)
        for s in held:
            np.savez_compressed(os.path.join(od, "SN%d.npz" % s), emb=E[s].astype(np.float16), y=Y[s])
        print("part %s: best val %.4f, held-out acc %.4f" % (part, best, FL.metrics(Y, P, held)["acc"]), flush=True)
        return
    import mmnet_core as C
    P, E = FL.predict(model, X, list(C.SUBS))
    od = os.path.join(EXT, "labram_ft_eog_all"); os.makedirs(od, exist_ok=True)
    for s in C.SUBS:
        np.savez_compressed(os.path.join(od, "SN%d.npz" % s), emb=E[s].astype(np.float16), y=Y[s])
    recs = ext_recs(groups)
    XE = {t: np.load(f)["x"][:, :6].astype(np.float16) for t, f in recs.items()}
    PE, EE = FL.predict(model, XE, list(recs))
    od = os.path.join(EXT, "emb", "fta_eog"); os.makedirs(od, exist_ok=True)
    for t in recs:
        np.savez_compressed(os.path.join(od, t + ".npz"), emb=EE[t].astype(np.float16), prob=PE[t].astype(np.float32))
        yt = np.load(recs[t])["y"]
        print("  %-12s encoder alone (one epoch, no HMM) acc %.4f" % (t, (PE[t].argmax(1) == yt).mean()), flush=True)
    torch.save(model.state_dict(), os.path.join(EXT, "labram_ft_eog_all.pt"))
    print("full encoder: best val %.4f" % best, flush=True)


def align(lam=0.01):
    import mmnet_core as C
    tr, va = all_split()
    part = cf_part_of(tr)
    od = os.path.join(DATA, "cbramod_emb", "labramfta_eog_all")
    os.makedirs(od, exist_ok=True)
    full = {s: np.load(os.path.join(EXT, "labram_ft_eog_all", "SN%d.npz" % s)) for s in C.SUBS}
    cf = {s: np.load(os.path.join(EXT, "labram_cf_eog_all", "SN%d.npz" % s))["emb"].astype(np.float64) for s in tr}
    maps = {}
    for p in range(3):
        rs = [s for s in tr if part[s] == p]
        Xc = np.concatenate([cf[s] for s in rs]); Yc = np.concatenate([full[s]["emb"] for s in rs]).astype(np.float64)
        mx, my = Xc.mean(0), Yc.mean(0)
        W = np.linalg.solve((Xc - mx).T @ (Xc - mx) + lam * len(Xc) * np.eye(Xc.shape[1]), (Xc - mx).T @ (Yc - my))
        maps[p] = (mx, my, W)
    for s in C.SUBS:
        if s in part:
            mx, my, W = maps[part[s]]
            e = (cf[s] - mx) @ W + my
        else:
            e = full[s]["emb"]
        np.savez_compressed(os.path.join(od, "SN%d.npz" % s), emb=np.asarray(e, np.float16), y=full[s]["y"])
    print("aligned embeddings ->", od)


# =========================================================================== MM-Net
def ahi_scale():
    """median over the ten iSLEEPS folds of the validation-patient AHI scale (final config)."""
    import mmnet_core as C
    from run_event_ahi import split as ev_split
    ks = []
    for fi, (tr_all, te) in enumerate(C.FOLDS):
        _, va = ev_split(fi, tr_all)
        d = np.load(os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "lesion", "seed_ensemble",
                                 "base-nocw-cross-ftlae-p8-al_fold%d.npz" % fi))
        r = []
        for s in va:
            st = C.hmm(d["A_log"], d["pi_log"], np.log(np.mean([d["SN%d|%d" % (s, sd)] for sd in S], 0) + C.EPS))
            ap = np.mean([d["A:SN%d|%d" % (s, sd)] for sd in S], 0)
            raw = ap[st != 0].sum() / max((st != 0).sum() * 30 / 3600, 0.5)
            ann = float(np.load(os.path.join(DATA, "aligned", "resp_events", "SN%d.npz" % s))["ahi_ann"])
            r.append(ann / max(raw, 1e-3))
        ks.append(np.median(r))
    return float(np.median(ks))


def mmnet(config, groups):
    import torch
    import run_lesion_aware as RL
    import cardio_cnn
    from run_event_ahi import metrics as ahi_metrics
    from sklearn.metrics import accuracy_score, average_precision_score, cohen_kappa_score, f1_score, roc_auc_score
    C, sweep, FINAL = RL.C, RL.sweep, RL.FINAL
    final = config.startswith("final")          # final7: the final architecture without nasal pressure
    n_ch = 8 if config == "final" else 7
    emb = "fta_eog" if final else "labram"
    arm = FINAL["arm"].replace("labram", "labramfta_eog_all") if final else FINAL["arm"]
    fusion = "cross" if final else "concat"
    orig_cw, orig_table, orig_nch = C.sqrt_cw, cardio_cnn.raw_cardio_table, cardio_cnn.N_CH
    if final:
        import run_mmnet_pflow as P
        C.sqrt_cw = lambda tr: torch.ones(5, dtype=torch.float32, device=C.DEV)
        if n_ch == 8:
            cardio_cnn.raw_cardio_table, cardio_cnn.N_CH = P.table_with_pressure, 8
    recs = ext_recs(groups)
    ext = {}
    for t, f in recs.items():
        d = np.load(f)
        fe = d["Feeg"].astype(np.float32); fe = (fe - fe.mean(0)) / (fe.std(0) + 1e-6)
        e = np.load(os.path.join(EXT, "emb", emb, t + ".npz"))["emb"].astype(np.float32)
        e = (e - e.mean(0)) / (e.std(0) + 1e-6)
        card = d["card"][:, :n_ch].astype(np.float32)
        card = (card - card.mean(axis=(0, 2), keepdims=True)) / (card.std(axis=(0, 2), keepdims=True) + 1e-6)
        ext[t] = (np.concatenate([fe, e], 1), card.reshape(len(card), -1), d["y"].astype(np.int64), d["apnea"].astype(np.int64))
    tr, va = all_split()
    Am = np.ones((C.NC, C.NC)); pi = np.ones(C.NC)
    probs = {t: [] for t in ext}; aprobs = {t: [] for t in ext}
    try:
        with sweep.config(C, arm, hidden=FINAL["hidden"], drop=FINAL["drop"], lr=FINAL["lr"], wd=FINAL["wd"],
                          cardio=FINAL["cardio"], cardio_mode=FINAL["cardio_mode"]) as h:
            for s in tr:
                y = C.DATA[s][2]; pi[y[0]] += 1
                for a_, b_ in zip(y[:-1], y[1:]): Am[a_, b_] += 1
            for sd in S:
                t0 = time.time()
                model = C.train_fold(tr, va, fusion, [], [], seed=sd, temporal=FINAL["temporal"])
                saved = dict(C.DATA)
                try:
                    C.DATA.update(ext)
                    for t in ext:
                        sp, ap = C.subj_infer(model, t, [], [])
                        probs[t].append(sp); aprobs[t].append(ap)
                finally:
                    C.DATA.clear(); C.DATA.update(saved)
                del model; torch.cuda.empty_cache()
                print("%s seed %d trained and applied [%.1f min]" % (config, sd, (time.time() - t0) / 60), flush=True)
    finally:
        C.sqrt_cw, cardio_cnn.raw_cardio_table, cardio_cnn.N_CH = orig_cw, orig_table, orig_nch
    A_log, pi_log = np.log(Am / Am.sum(1, keepdims=True)), np.log(pi / pi.sum())
    k = ahi_scale()
    os.makedirs(RES, exist_ok=True)
    np.savez_compressed(os.path.join(RES, "isruc_%s_probs.npz" % config), A_log=A_log, pi_log=pi_log,
                        **{"%s|%d" % (t, sd): probs[t][i] for t in ext for i, sd in enumerate(S)},
                        **{"A:%s|%d" % (t, sd): aprobs[t][i] for t in ext for i, sd in enumerate(S)})
    out = {"config": config, "ahi_scale_k": k}
    lines = ["ISRUC, MM-Net %s trained on all 99 iSLEEPS patients, applied unchanged (AHI scale k %.3f from iSLEEPS)" % (config, k)]
    for g in groups + (["+".join(groups)] if len(groups) > 1 else []):
        ts = [t for t in ext if t.split("_")[0] in g.split("+")]
        yt = np.concatenate([ext[t][2] for t in ts])
        row = {}
        for name, members in [("seed %d" % sd, [i]) for i, sd in enumerate(S)] + [("3-seed", [0, 1, 2])]:
            yp = np.concatenate([C.hmm(A_log, pi_log, np.log(np.mean([probs[t][i] for i in members], 0) + C.EPS)) for t in ts])
            row[name] = dict(acc=float(accuracy_score(yt, yp)), mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                             kappa=float(cohen_kappa_score(yt, yp)))
        single = np.mean([row["seed %d" % sd]["acc"] for sd in S])
        at = np.concatenate([ext[t][3] for t in ts]); ap3 = np.concatenate([np.mean(aprobs[t], 0) for t in ts])
        auc = float(roc_auc_score(at, ap3)) if len(np.unique(at)) > 1 else None
        apv = float(average_precision_score(at, ap3)) if auc is not None else None
        ahi_true, ahi_pred = [], []
        for t in ts:
            d = np.load(recs[t]); y = d["y"]; nev = d["n_events"]
            ahi_true.append(nev[y != 0].sum() / max((y != 0).sum() * 30 / 3600, 1e-6))
            st = C.hmm(A_log, pi_log, np.log(np.mean(probs[t], 0) + C.EPS))
            apm = np.mean(aprobs[t], 0)
            ahi_pred.append(k * apm[st != 0].sum() / max((st != 0).sum() * 30 / 3600, 0.5))
        am = ahi_metrics(ahi_pred, ahi_true) if len(ts) > 2 else {}
        out[g] = dict(n=len(ts), epochs=int(len(yt)), staging=row, single_acc=float(single), resp_auc=auc, resp_ap=apv,
                      ahi=am, ahi_true={t: float(v) for t, v in zip(ts, ahi_true)}, ahi_pred={t: float(v) for t, v in zip(ts, ahi_pred)})
        lines.append("%-8s %2d rec  staging single %.4f | 3-seed acc %.4f mf1 %.4f kappa %.4f | resp AUC %s AP %s | AHI %s"
                     % (g, len(ts), single, row["3-seed"]["acc"], row["3-seed"]["mf1"], row["3-seed"]["kappa"],
                        "%.3f" % auc if auc else "n/a", "%.3f" % apv if apv else "n/a",
                        ("rho %.3f MAE %.1f sev acc %.2f kappa %.2f" % (am["rho"], am["mae"], am["sev_acc"], am["sev_kappa"])) if am else "n/a"))
    json.dump(out, open(os.path.join(RES, "isruc_%s.json" % config), "w"), indent=1)
    open(os.path.join(RES, "isruc_%s.txt" % config), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "labram", "align", "mmnet"])
    ap.add_argument("--group", default="III")
    ap.add_argument("--groups", nargs="+", default=["II", "III"])
    ap.add_argument("--part", default="full")
    ap.add_argument("--config", default="final", choices=["final", "final7", "frozen"])
    a = ap.parse_args()
    if a.stage == "build":
        build(a.group)
    elif a.stage == "labram":
        labram(a.part, a.groups)
    elif a.stage == "align":
        align()
    else:
        mmnet(a.config, a.groups)
