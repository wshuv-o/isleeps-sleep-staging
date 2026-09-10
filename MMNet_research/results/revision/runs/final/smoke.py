import sys, os, glob, time
sys.path.insert(0, r"D:\proc\isleeps-sleep-staging\MMNet_research\foundation")
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
import run_yasa
f = sorted(glob.glob(r"D:\proc\isleeps-sleep-staging\data\processed7\SN*.npz"))[0]
t0 = time.time()
y, yp = run_yasa.stage_one(f)
print("file", os.path.basename(f), "epochs", len(y), "in %.1f s" % (time.time()-t0))
print("acc %.4f  kappa %.4f" % (accuracy_score(y, yp), cohen_kappa_score(y, yp)))
print("true", np.bincount(y, minlength=5).tolist())
print("pred", np.bincount(yp, minlength=5).tolist())
