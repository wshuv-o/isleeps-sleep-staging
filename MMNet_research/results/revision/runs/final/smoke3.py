import time, glob, warnings, numpy as np, mne
warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")
from yasa import SleepStaging
from sklearn.metrics import accuracy_score, cohen_kappa_score

f = sorted(glob.glob(r"D:\proc\isleeps-sleep-staging\data\processed7\SN*.npz"))[0]
d = np.load(f, allow_pickle=True); x = d["x"]; y = d["y"].astype(int); n = len(y)
# YASA reads exactly three channels; feeding it all seven filters four for nothing
KEEP = [0, 4, 6]                      # C4:M1, E1:M2, EMG
NAMES = ["C4:M1", "E1:M2", "EMG"]
t0 = time.time()
cont = np.asarray(x[:, KEEP], dtype=np.float64).transpose(1, 0, 2).reshape(3, n*3000) * 1e-6
info = mne.create_info(NAMES, 100, ch_types=["eeg", "eog", "emg"])
raw = mne.io.RawArray(cont, info, verbose=False)
print("raw built in %.1f s, %.0f MB" % (time.time()-t0, cont.nbytes/1e6), flush=True)
t1 = time.time()
sls = SleepStaging(raw, eeg_name="C4:M1", eog_name="E1:M2", emg_name="EMG")
pred = sls.predict()
print("staged in %.1f s" % (time.time()-t1), flush=True)
M = {"W":0,"N1":1,"N2":2,"N3":3,"R":4}
yp = np.array([M.get(str(s),0) for s in pred])[:n]
print("acc %.4f  kappa %.4f" % (accuracy_score(y[:len(yp)], yp), cohen_kappa_score(y[:len(yp)], yp)))
print("true", np.bincount(y, minlength=5).tolist())
print("pred", np.bincount(yp, minlength=5).tolist())
