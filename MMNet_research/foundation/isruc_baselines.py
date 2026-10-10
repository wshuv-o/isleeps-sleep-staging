"""Every reimplemented baseline on ISRUC-Sleep subgroup I, on MM-Net's ISRUC folds.

compare_baselines.py sets MM-Net against each baseline fold by fold on iSLEEPS. The ISRUC-I
cross-validation (isruc_cv.py --groups I --folds 10) gives MM-Net ten fold values on a second
corpus, but no baseline had been run there, so the comparison stopped at one cohort. This runs
the same baselines on the same ten ISRUC-I folds so the paired comparison can be repeated.

As in run_healthy_baselines.py, each baseline's own `run_fold` (and with it its model, loss,
optimiser, early stopping and decoding) is imported from its script; nothing is reimplemented.
Only two things change:

  data    H:/isleeps_ext/isruc/I/I_S<n>.npz (external_final.py build), mapped to the arrays each
          script reads on iSLEEPS. x is already in processed7 channel order (C4-M1, C3-M2, O2-M1,
          O1-M2, E1, E2, chin EMG) and card in data/multimodal order (ECG, Flow, Thorax, Abdomen,
          Effort, SpO2, Pulse) plus nasal pressure at index 7, so every script keeps its channel
          indices and its normalisation:
            AttnSleep, TinySleepNet, U-Time, 1D-ResNet-SE-LSTM, AT-BiLSTM   C4-M1, raw 100 Hz
            SleepTransformer                                                C4-M1 spectrogram
            LSTM (corpus reimpl.)                                           C4-M1 + E1
            Huttunen et al.     SpO2, Flow (thermistor), Effort, C4-M1 at 32 Hz (huttunen.json);
                                huttunen5 adds nasal pressure (huttunen5.json)
            StagingSeqNet       C4-M1, O2-M1, E1, chin EMG (zero-shot, as on iSLEEPS)
  folds   isruc_cv.folds(["I"], 10): train / validation / test recordings per fold. Validation is
          what each script early-stops on, test is only scored. The baselines' own iSLEEPS
          validation rules (10 patients, max(5, n // 9), ...) are replaced by these validation
          recordings so every model sees exactly MM-Net's split.

Seeds are the ones each baseline's iSLEEPS JSON was produced with (42/1/7 where the script is
seeded three times, 42 otherwise). StagingSeqNet is never trained on the target corpus: it is
pretrained on Sleep-EDF (4 channels, built once from the raw EDFs with sleepedf_full_preprocess,
cached in H:/isleeps_ext/sleep_edf_full_proc) once per seed and scored on each fold's test
recordings, which is the protocol its iSLEEPS row follows.

Metrics per (baseline, seed, fold): accuracy, macro-F1 and Cohen's kappa pooled over the fold's
test recordings, with the decoding the script uses on iSLEEPS (train-fold HMM Viterbi for
AttnSleep and the LSTM, argmax otherwise). Each run is written as it finishes (atomic replace)
to results/revision/runs/lesion/external/isruc_I_baselines.json, with its confusion matrix, and
the run is skipped on restart. Per seed: ten fold values ("per_seed", the layout
compare_baselines reads) and totals pooled over all 100 test recordings; per fold: the seed mean.
Predicted stages per test recording go to isruc_I_baselines_pred/<baseline>/s<seed>_f<k>.npz.

    G:/codes/isleeps/prepenv/Scripts/python isruc_baselines.py <name|all> [--smoke]

  names: attnsleep lstm resnetse atbilstm sleeptransformer tinysleepnet utime huttunen
         stagingseqnet huttunen5 (all = every one, in that order)
  --smoke  CPU only (CUDA hidden), fold 0, 3 training / 1 validation / 2 test recordings, one
           epoch; results go to a temporary file, never to the real JSON.

AttnSleep needs braindecode, which imports only in the prep venv (system torchaudio does not
load), so the venv's python is used for every baseline.
"""
import argparse
import glob
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EXT = "H:/isleeps_ext"
RES = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "lesion", "external")
OUT = os.path.join(RES, "isruc_I_baselines.json")
SEDF4 = os.path.join(EXT, "sleep_edf_full_proc")          # 4-channel Sleep-EDF for StagingSeqNet
SEDF_RAW = os.path.join(REPO, "data", "sleep_edf")
SEDF3 = os.path.join(REPO, "data", "sleep_edf_proc")      # 3-channel cache, same epochs: a check
GROUPS, N_FOLDS = ["I"], 10

# display name, script, iSLEEPS JSON, seeds, input, decoding
SPEC = {
    "attnsleep": ("AttnSleep", "run_attnsleep_seeded.py", "attnsleep_seeded.json", [42, 1, 7],
                  "C4-M1, per-epoch z-score", "train-fold HMM Viterbi"),
    "lstm": ("LSTM (corpus, reimpl.)", "run_isleeps_lstm.py", "isleeps_lstm.json", [42, 1, 7],
             "C4-M1 + E1, per-epoch z-score, 5-epoch context", "train-fold HMM Viterbi"),
    "resnetse": ("1D-ResNet-SE-LSTM", "run_recent_baselines.py resnetse", "resnetse_stroke.json",
                 [42], "C4-M1, per-recording z-score, 20-epoch sequences", "argmax"),
    "atbilstm": ("AT-BiLSTM", "run_recent_baselines.py atbilstm", "atbilstm_stroke.json", [42],
                 "C4-M1, per-recording z-score, 20-epoch sequences", "argmax"),
    "sleeptransformer": ("SleepTransformer", "run_sleeptransformer.py", "sleeptransformer.json",
                         [42], "C4-M1 log spectrogram, per-recording per-bin standardised",
                         "argmax"),
    "tinysleepnet": ("TinySleepNet", "run_tinysleepnet.py", "tinysleepnet.json", [42],
                     "C4-M1, per-recording z-score, 15-epoch sequences", "argmax"),
    "utime": ("U-Time", "run_utime.py", "utime_stroke.json", [42],
              "C4-M1, per-recording z-score, 32-epoch segments", "argmax"),
    "huttunen": ("Huttunen et al.", "run_huttunen.py", "huttunen.json", [42, 1, 7],
                 "SpO2, Flow (thermistor), Effort, C4-M1 at 32 Hz", "argmax"),
    "stagingseqnet": ("StagingSeqNet", "run_stagingseqnet_seeded.py", "stagingseqnet_seeded.json",
                      [42, 1, 7], "C4-M1, O2-M1, E1, chin EMG; pretrained on Sleep-EDF only "
                      "(zero-shot)", "argmax"),
    "huttunen5": ("Huttunen et al. (5 signals)", "run_huttunen.py, HUTTUNEN_SIGNALS=5",
                  "huttunen5.json", [42, 1, 7],
                  "SpO2, Flow (thermistor), Effort, nasal pressure, C4-M1 at 32 Hz", "argmax"),
}
ORDER = list(SPEC)

ap = argparse.ArgumentParser()
ap.add_argument("name", choices=ORDER + ["all"])
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--out", default=None)
A = ap.parse_args()
if A.smoke:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # before torch is imported ("" is unset on Windows)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix  # noqa

import isruc_cv                                             # noqa: E402

if A.smoke:
    torch.set_num_threads(4)                                # leave the CPU to the running chain
    SPEC = {k: v[:3] + (v[3][:1],) + v[4:] for k, v in SPEC.items()}   # first seed only
    OUT = A.out or os.path.join(tempfile.gettempdir(), "isruc_I_baselines_smoke.json")
else:
    OUT = A.out or OUT
PRED = os.path.splitext(OUT)[0] + "_pred"


# --------------------------------------------------------------------------- bookkeeping
def metrics(yt, yp):
    return dict(n=int(len(yt)), acc=float(accuracy_score(yt, yp)),
                mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                kappa=float(cohen_kappa_score(yt, yp)))


def from_confusion(cm):
    """acc, macro-F1 (classes present in truth or prediction, as sklearn) and kappa."""
    cm = np.asarray(cm, float); n = cm.sum(); tp = np.diag(cm)
    rows, cols = cm.sum(1), cm.sum(0)
    present = (rows + cols) > 0
    f1 = 2 * tp[present] / (rows[present] + cols[present])
    po, pe = tp.sum() / n, (rows * cols).sum() / n ** 2
    return dict(n=int(n), acc=float(po), mf1=float(f1.mean()),
                kappa=float((po - pe) / (1 - pe)) if pe < 1 else 0.0)


def load_json():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def save_json(res):
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, OUT)


def record(key, folds, run_key, run, yt, yp, te, lens):
    """Merge one finished run into the JSON (re-read first, so other baselines are kept)."""
    label, script, src, seeds, inp, dec = SPEC[key]
    od = os.path.join(PRED, key); os.makedirs(od, exist_ok=True)
    seed, fi = run_key.split("|")
    np.savez_compressed(os.path.join(od, "s%s_f%s.npz" % (seed, fi)), tags=np.array(te),
                        lens=np.array(lens), y=yt.astype(np.int8), pred=yp.astype(np.int8))
    res = load_json()
    res.setdefault("protocol", "ISRUC-Sleep subgroup I, isruc_cv.folds(['I'], 10): "
                   "subject-independent, validation recordings for early stopping, test "
                   "recordings scored only")
    res.setdefault("folds", [dict(fold=k, train=tr, val=va, test=tt)
                             for k, (tr, va, tt) in enumerate(folds)])
    b = res.setdefault("baselines", {}).setdefault(label, {})
    b.update(name=key, script=script, isleeps_json=src, seeds=seeds, input=inp, decoding=dec)
    b.setdefault("runs", {})[run_key] = run
    summarise(b, len(folds))
    save_json(res)


def summarise(b, nf):
    runs, seeds = b["runs"], b["seeds"]
    per_seed, pooled = {}, {}
    for s in seeds:
        rs = [runs.get("%d|%d" % (s, k)) for k in range(nf)]
        if all(rs):
            per_seed["seed|%d" % s] = {m: [r[m] for r in rs] for m in ("acc", "mf1", "kappa")}
            pooled["seed|%d" % s] = from_confusion(sum(np.array(r["confusion"]) for r in rs))
    b["per_seed"], b["pooled"] = per_seed, pooled
    b["complete"] = len(per_seed) == len(seeds)
    if b["complete"]:
        b["fold_mean"] = {m: list(np.mean([per_seed[k][m] for k in per_seed], 0))
                          for m in ("acc", "mf1", "kappa")}
        b["pooled"]["seed_mean"] = {m: float(np.mean([pooled[k][m] for k in per_seed]))
                                    for m in ("acc", "mf1", "kappa")}
        b["fold_summary"] = {m: dict(mean=float(np.mean(b["fold_mean"][m])),
                                     sd=float(np.std(b["fold_mean"][m], ddof=1)) if nf > 1 else 0.0)
                             for m in ("acc", "mf1", "kappa")}
    b["minutes"] = float(sum(r.get("minutes", 0.0) for r in runs.values()))


# --------------------------------------------------------------------------- ISRUC data
def raw(tags):
    R = isruc_cv.recs(GROUPS)
    for t in tags:
        yield t, np.load(R[t])


def load_c4(tags, per_epoch):
    """C4-M1 [n, 3000] float32: per-epoch z-score (AttnSleep) or per-recording (the others)."""
    data = {}
    for t, d in raw(tags):
        x = np.asarray(d["x"][:, 0], dtype=np.float32)
        if per_epoch:
            x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
        else:
            x = (x - x.mean()) / (x.std() + 1e-6)
        data[t] = (x, d["y"].astype(np.int64))
    return data


def load_lstm(tags, mod):
    data = {}
    for t, d in raw(tags):
        x = np.asarray(d["x"][:, [mod.EEG_CH, mod.EOG_CH]], dtype=np.float32)
        x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)
        data[t] = (x, d["y"].astype(np.int64))
    return data


def load_spec(tags, ST):
    data = {}
    for t, d in raw(tags):
        x = torch.tensor(np.asarray(d["x"][:, ST.EEG_CH], dtype=np.float32), device=ST.DEV)
        s = ST.spectrogram(x)
        s = (s - s.mean((0, 1), keepdim=True)) / (s.std((0, 1), keepdim=True) + 1e-6)
        data[t] = (s.half().cpu(), d["y"].astype(np.int64))
    return data


def load_huttunen(tags, H, five):
    from scipy.signal import resample_poly
    data = {}
    for t, d in raw(tags):
        card = d["card"].astype(np.float32)                       # [n, 8, 750] at 25 Hz
        c = card[:, [H.CARD_SPO2, H.CARD_FLOW, H.CARD_EFFORT] + ([7] if five else []), :]
        c32 = resample_poly(c, H.FS, 25, axis=-1).astype(np.float32)
        e32 = resample_poly(np.asarray(d["x"][:, [H.EEG_C4]], dtype=np.float32), H.FS, 100,
                            axis=-1).astype(np.float32)
        x = np.concatenate([c32, e32], axis=1)
        assert x.shape[-1] == H.SAMP, x.shape
        mu = x.mean(axis=(0, 2), keepdims=True); sd = x.std(axis=(0, 2), keepdims=True) + 1e-6
        data[t] = ((x - mu) / sd, d["y"].astype(np.int64), d["apnea"].astype(np.float32))
    return data


# --------------------------------------------------------------------------- one fold per baseline
def huttunen_fold(H, data, tr, va, te, seed):
    """run_huttunen.run_fold returns metrics only; its last seqs() / infer() calls are the test
    set, so they are recorded on the way through to keep the predictions. sqrt_cw reads
    mmnet_core.DATA (iSLEEPS), so it is pointed at these labels for the call."""
    last = {}
    seqs0, infer0, cw0 = H.seqs, H.infer, H.C.sqrt_cw

    def seqs(d, subs):
        out = seqs0(d, subs); last["seqs"] = (list(subs), out); return out

    def infer(model, X, bs=8):
        out = infer0(model, X, bs); last["infer"] = out; return out

    def sqrt_cw(subs):
        cc = np.zeros(5, np.int64)
        for s in subs:
            cc += np.bincount(data[s][1], minlength=5)
        inv = np.sqrt(cc.sum() / (5 * np.maximum(cc, 1)))
        return torch.tensor(inv / inv.mean(), dtype=torch.float32, device=H.DEV)

    H.seqs, H.infer, H.C.sqrt_cw = seqs, infer, sqrt_cw
    try:
        r = H.run_fold(data, tr, va, te, seed)
    finally:
        H.seqs, H.infer, H.C.sqrt_cw = seqs0, infer0, cw0
    assert last["seqs"][0] == list(te)
    _, Yte, _, Mte = last["seqs"][1]
    sp, _ = last["infer"]
    mk = Mte.numpy().reshape(-1) > 0
    yt, yp = Yte.numpy().reshape(-1)[mk], sp.reshape(-1, 5).argmax(1)[mk]
    assert abs(accuracy_score(yt, yp) - r["acc"]) < 1e-9
    return yt, yp, dict(resp_auc=r["auc"], resp_ap=r["ap"])


def run_trained(key, folds, smoke):
    """Every baseline that is trained on the target corpus: one call to its run_fold per run."""
    label, _, _, seeds, _, _ = SPEC[key]
    if key == "attnsleep":
        import run_attnsleep_seeded as M
        if smoke: M.EPOCHS = 1
        load = lambda tags: load_c4(tags, per_epoch=True)
        fold = lambda d, tr, va, te, s: M.run_fold(d, tr, va, te, s)
    elif key == "lstm":
        import run_isleeps_lstm as M
        if smoke: M.EPOCHS = 1
        load = lambda tags: load_lstm(tags, M)
        fold = lambda d, tr, va, te, s: M.run_fold(d, tr, va, te, s)
    elif key in ("resnetse", "atbilstm"):
        import run_recent_baselines as M
        load = lambda tags: load_c4(tags, per_epoch=False)
        fold = lambda d, tr, va, te, s: M.run_fold(key, d, tr, va, te, s,
                                                   **(dict(epochs=1) if smoke else {}))
    elif key == "sleeptransformer":
        import run_sleeptransformer as M
        if smoke: M.EPOCHS = 1
        load = lambda tags: load_spec(tags, M)
        fold = lambda d, tr, va, te, s: M.run_fold(d, tr, va, te, s)
    elif key == "tinysleepnet":
        import run_tinysleepnet as M
        if smoke: M.EPOCHS = 1
        load = lambda tags: load_c4(tags, per_epoch=False)
        fold = lambda d, tr, va, te, s: M.run_fold(d, tr, va, te, s)
    elif key == "utime":
        import run_utime as M
        if smoke: M.EPOCHS = 1
        load = lambda tags: load_c4(tags, per_epoch=False)
        fold = lambda d, tr, va, te, s: M.run_fold(d, tr, va, te, s)
    elif key in ("huttunen", "huttunen5"):
        import run_huttunen as M
        if smoke: M.EPOCHS = 1
        load = lambda tags: load_huttunen(tags, M, five=(key == "huttunen5"))
        fold = lambda d, tr, va, te, s: huttunen_fold(M, d, tr, va, te, s)
    else:
        raise ValueError(key)

    done = load_json().get("baselines", {}).get(label, {}).get("runs", {})
    todo = [(s, k) for s in seeds for k in range(len(folds)) if "%d|%d" % (s, k) not in done]
    if not todo:
        print("[skip] %s: all %d runs present" % (label, len(seeds) * len(folds)), flush=True)
        return
    t0 = time.time()
    data = load(sorted({t for f in folds for part in f for t in part}))
    print("%s on ISRUC-I: %d recordings, %d epochs, %d runs to do [load %.1f min]"
          % (label, len(data), sum(len(v[1]) for v in data.values()), len(todo),
             (time.time() - t0) / 60), flush=True)
    for s, k in todo:
        tr, va, te = folds[k]
        t1 = time.time()
        out = fold(data, tr, va, te, s)
        yt, yp = np.asarray(out[0]), np.asarray(out[1])
        lens = [len(data[t][1]) for t in te]
        assert len(yt) == sum(lens) and np.array_equal(yt, np.concatenate([data[t][1] for t in te]))
        run = dict(seed=s, fold=k, **metrics(yt, yp),
                   confusion=confusion_matrix(yt, yp, labels=list(range(5))).tolist(),
                   minutes=(time.time() - t1) / 60)
        if len(out) > 2:
            run.update(out[2])
        record(key, folds, "%d|%d" % (s, k), run, yt, yp, te, lens)
        print("  %s seed %d fold %d  %5d ep  acc %.4f  mF1 %.4f  kappa %.4f  [%.1f min]"
              % (label, s, k, run["n"], run["acc"], run["mf1"], run["kappa"],
                 (time.time() - t0) / 60), flush=True)
        if not smoke:
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- StagingSeqNet
def build_sedf4(dst, recs=None):
    """4-channel Sleep-EDF (Fpz-Cz, Pz-Oz, EOG horizontal, EMG submental) with the repo's own
    sleepedf_full_preprocess.process, which is what StagingSeqNet was pretrained on. This
    machine's data/sleep_edf_proc holds only the first three, on the same epochs."""
    sys.path.insert(0, os.path.join(REPO, "HAGNet_research", "train", "sleepedf"))
    import sleepedf_full_preprocess as SP
    os.makedirs(dst, exist_ok=True)
    psgs = sorted(glob.glob(os.path.join(SEDF_RAW, "SC*-PSG.edf")))
    if recs is not None:
        psgs = [p for p in psgs if os.path.basename(p)[:6] in recs]
    for p in psgs:
        rec = os.path.basename(p)[:6]
        out = os.path.join(dst, rec + ".npz")
        if os.path.exists(out):
            continue
        hyp = glob.glob(os.path.join(SEDF_RAW, rec + "*-Hypnogram.edf"))
        if not hyp:
            continue
        x, y = SP.process(p, hyp[0])
        ref = os.path.join(SEDF3, rec + ".npz")
        if os.path.exists(ref):                     # same epochs as the 3-channel cache
            y3 = np.load(ref)["y"]
            if not np.array_equal(y3, y):
                raise RuntimeError("%s: 4-channel epochs differ from %s" % (rec, ref))
        np.savez_compressed(out + ".tmp.npz", x=x, y=y)
        os.replace(out + ".tmp.npz", out)
        print("  [sedf4] %s %d ep" % (rec, len(y)), flush=True)


def run_stagingseqnet(folds, smoke):
    key = "stagingseqnet"
    label, _, _, seeds, _, _ = SPEC[key]
    sedf_dir = SEDF4
    if smoke:
        sedf_dir = os.path.join(tempfile.gettempdir(), "isruc_smoke_sedf4")
    done = load_json().get("baselines", {}).get(label, {}).get("runs", {})
    todo = {s: [k for k in range(len(folds)) if "%d|%d" % (s, k) not in done] for s in seeds}
    if not any(todo.values()):
        print("[skip] %s: all runs present" % label, flush=True)
        return
    t0 = time.time()
    build_sedf4(sedf_dir, recs={"SC4001", "SC4002"} if smoke else None)
    import run_stagingseqnet_seeded as S
    T = S.T
    T.SEDF = sedf_dir
    if smoke:
        S.PRE_EPOCHS = 1
    sedf = T.sedf_list()
    for r in sedf:
        T.load_sedf(r)
    n_sedf = sum(len(T.DATA[("s", r)][1]) for r in sedf)
    print("StagingSeqNet: pretraining on %d Sleep-EDF recordings (%d epochs)%s"
          % (len(sedf), n_sedf, "" if smoke or n_sedf == 43330 else
             "  WARNING: iSLEEPS run pretrained on 43330"), flush=True)
    sedf_k = [("s", r) for r in sedf]
    need = sorted({t for s in seeds for k in todo[s] for t in folds[k][2]})
    items = {t: (S.norm(np.asarray(d["x"][:, T.ISL_COLS], dtype=np.float32)),
                 d["y"].astype(np.int64)) for t, d in raw(need)}
    for s in seeds:
        if not todo[s]:
            continue
        t1 = time.time()
        print("  pretrain seed %d" % s, flush=True)
        model = S.pretrain(s, sedf_k)
        pre_min = (time.time() - t1) / 60
        for k in todo[s]:
            te = folds[k][2]
            yp = np.concatenate([S.predict(model, items[t][0]) for t in te])
            yt = np.concatenate([items[t][1] for t in te])
            lens = [len(items[t][1]) for t in te]
            run = dict(seed=s, fold=k, **metrics(yt, yp),
                       confusion=confusion_matrix(yt, yp, labels=list(range(5))).tolist(),
                       minutes=pre_min / len(todo[s]))
            record(key, folds, "%d|%d" % (s, k), run, yt, yp, te, lens)
            print("  %s seed %d fold %d  %5d ep  acc %.4f  mF1 %.4f  kappa %.4f  [%.1f min]"
                  % (label, s, k, run["n"], run["acc"], run["mf1"], run["kappa"],
                     (time.time() - t0) / 60), flush=True)
        del model
        if not smoke:
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- main
def main():
    folds = isruc_cv.folds(GROUPS, N_FOLDS)
    if A.smoke:
        tr, va, te = folds[0]
        va = [t for t in va if t not in tr][:1] or va[:1]
        folds = [(tr[:3], va, te[:2])]
        print("SMOKE (CPU, fold 0 subset): train %s val %s test %s -> %s"
              % (folds[0][0], folds[0][1], folds[0][2], OUT), flush=True)
    print("device: %s" % ("cuda" if torch.cuda.is_available() else "cpu"), flush=True)
    if A.smoke:
        assert not torch.cuda.is_available(), "smoke must not see the GPU"
    names = ORDER if A.name == "all" else [A.name]
    for key in names:
        t0 = time.time()
        if key == "stagingseqnet":
            run_stagingseqnet(folds, A.smoke)
        else:
            run_trained(key, folds, A.smoke)
        b = load_json().get("baselines", {}).get(SPEC[key][0], {})
        if b.get("complete"):
            fs, pm = b["fold_summary"], b["pooled"]["seed_mean"]
            print("%s DONE  fold mean acc %.4f +- %.4f  mF1 %.4f  kappa %.4f | pooled acc %.4f "
                  "mF1 %.4f kappa %.4f  [%.1f min]"
                  % (SPEC[key][0], fs["acc"]["mean"], fs["acc"]["sd"], fs["mf1"]["mean"],
                     fs["kappa"]["mean"], pm["acc"], pm["mf1"], pm["kappa"],
                     (time.time() - t0) / 60), flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
