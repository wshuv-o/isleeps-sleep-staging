# NO sklearn in this process: importing it alongside yasa/LightGBM deadlocks on
# the OpenMP runtime (the libiomp5md.dll conflict this repo already documents).
# Predictions are saved raw and scored by a separate process.
import time, os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, mne
mne.set_log_level("ERROR")
from yasa import SleepStaging

P7 = r"D:\proc\isleeps-sleep-staging\data\processed7"
OUT = r"D:\proc\isleeps-sleep-staging\MMNet_research\results\revision\runs\final\yasa_preds.npz"
M = {"W":0,"N1":1,"N2":2,"N3":3,"R":4}
files = sorted(glob.glob(os.path.join(P7,"SN*.npz")), key=lambda p:int(os.path.basename(p)[2:-4]))
print("%d recordings" % len(files), flush=True)

yt, yp, rec = [], [], []
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
        p = np.array([M.get(str(s),0) for s in pred], dtype=np.int64)
        k = min(len(p), n)
        yt.append(y[:k]); yp.append(p[:k]); rec.append(np.full(k, sid))
        acc = float((y[:k]==p[:k]).mean())
    except Exception as e:
        print("[FAIL] %-6s %s: %s" % (sid, type(e).__name__, e), flush=True); continue
    print("%3d/%d %-6s %4d ep acc %.3f [%.0fs]" % (i,len(files),sid,k,acc,time.time()-t0), flush=True)

np.savez_compressed(OUT, y_true=np.concatenate(yt), y_pred=np.concatenate(yp),
                    recording=np.concatenate(rec))
print("saved %s in %.1f min" % (OUT, (time.time()-t0)/60), flush=True)
