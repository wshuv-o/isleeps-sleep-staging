import time, os, glob, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, mne
mne.set_log_level("ERROR")
from yasa import SleepStaging
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix

P7 = r"D:\proc\isleeps-sleep-staging\data\processed7"
OUT = r"D:\proc\isleeps-sleep-staging\MMNet_research\results\revision\runs\final\yasa_zeroshot.json"
M = {"W":0,"N1":1,"N2":2,"N3":3,"R":4}
S = ["W","N1","N2","N3","R"]
files = sorted(glob.glob(os.path.join(P7,"SN*.npz")), key=lambda p:int(os.path.basename(p)[2:-4]))
print("%d recordings" % len(files), flush=True)

per, YT, YP = {}, [], []
t0 = time.time()
for i, f in enumerate(files, 1):
    sid = os.path.basename(f)[:-4]
    try:
        d = np.load(f, allow_pickle=True)
        y = d["y"].astype(np.int64); n = len(y)
        cont = np.asarray(d["x"][:, [0,4,6]], dtype=np.float64).transpose(1,0,2).reshape(3, n*3000) * 1e-6
        info = mne.create_info(["C4:M1","E1:M2","EMG"], 100, ch_types=["eeg","eog","emg"])
        raw = mne.io.RawArray(cont, info, verbose=False)
        pred = SleepStaging(raw, eeg_name="C4:M1", eog_name="E1:M2", emg_name="EMG").predict()
        yp = np.array([M.get(str(s),0) for s in pred], dtype=np.int64)
        k = min(len(yp), n); y, yp = y[:k], yp[:k]
    except Exception as e:
        print("[FAIL] %-6s %s: %s" % (sid, type(e).__name__, e), flush=True); continue
    per[sid] = dict(n=int(len(y)), acc=float(accuracy_score(y,yp)), kappa=float(cohen_kappa_score(y,yp)))
    YT.append(y); YP.append(yp)
    print("%3d/%d %-6s %4d ep acc %.3f [%.0fs]" % (i,len(files),sid,len(y),per[sid]["acc"],time.time()-t0), flush=True)

YT, YP = np.concatenate(YT), np.concatenate(YP)
cm = confusion_matrix(YT, YP, labels=range(5), normalize="true")
res = dict(model="YASA", version=__import__("yasa").__version__,
    protocol="zero-shot, pretrained weights, no metadata supplied",
    n_recordings=len(per), n_epochs=int(len(YT)),
    acc=float(accuracy_score(YT,YP)), mf1=float(f1_score(YT,YP,average="macro",zero_division=0)),
    kappa=float(cohen_kappa_score(YT,YP)),
    per_class_recall={S[i]: float(cm[i,i]) for i in range(5)},
    predicted_wake_fraction=float((YP==0).mean()), true_wake_fraction=float((YT==0).mean()),
    per_recording=per,
    acc_sd_across_recordings=float(np.std([v["acc"] for v in per.values()], ddof=1)),
    minutes=(time.time()-t0)/60)
json.dump(res, open(OUT,"w"), indent=1)
print("\nYASA acc %.4f mF1 %.4f kappa %.4f" % (res["acc"],res["mf1"],res["kappa"]), flush=True)
print("recall:", {k: round(v,3) for k,v in res["per_class_recall"].items()}, flush=True)
print("Wake predicted %.1f%% vs true %.1f%%" % (100*res["predicted_wake_fraction"],100*res["true_wake_fraction"]), flush=True)
