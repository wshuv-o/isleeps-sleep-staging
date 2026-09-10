import sys, os, glob, time, json, warnings
sys.path.insert(0, r"D:\proc\isleeps-sleep-staging\MMNet_research\foundation")
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix
import run_yasa

files = sorted(glob.glob(r"D:\proc\isleeps-sleep-staging\data\processed7\SN*.npz"),
               key=lambda p: int(os.path.basename(p)[2:-4]))
print("found %d recordings" % len(files), flush=True)
per, YT, YP = {}, [], []
t0 = time.time()
for i, f in enumerate(files, 1):
    sid = os.path.basename(f)[:-4]
    try:
        y, yp = run_yasa.stage_one(f)
    except Exception as e:
        print("[FAIL] %-6s %s: %s" % (sid, type(e).__name__, e), flush=True); continue
    per[sid] = dict(n=int(len(y)), acc=float(accuracy_score(y, yp)),
                    kappa=float(cohen_kappa_score(y, yp)))
    YT.append(y); YP.append(yp)
    print("%3d/%d %-6s %4d ep  acc %.3f  [%.1f s]" % (i, len(files), sid, len(y),
          per[sid]["acc"], time.time()-t0), flush=True)
YT, YP = np.concatenate(YT), np.concatenate(YP)
cm = confusion_matrix(YT, YP, labels=range(5), normalize="true")
S = ["W","N1","N2","N3","R"]
res = dict(model="YASA", version=__import__("yasa").__version__,
           protocol="zero-shot, pretrained weights, no metadata supplied",
           n_recordings=len(per), n_epochs=int(len(YT)),
           acc=float(accuracy_score(YT, YP)),
           mf1=float(f1_score(YT, YP, average="macro", zero_division=0)),
           kappa=float(cohen_kappa_score(YT, YP)),
           per_class_recall={S[i]: float(cm[i,i]) for i in range(5)},
           predicted_wake_fraction=float((YP==0).mean()),
           true_wake_fraction=float((YT==0).mean()),
           per_recording=per,
           acc_sd_across_recordings=float(np.std([v["acc"] for v in per.values()], ddof=1)),
           minutes=(time.time()-t0)/60)
json.dump(res, open(r"D:\proc\isleeps-sleep-staging\MMNet_research\results\revision\runs\final\yasa_zeroshot.json","w"), indent=1)
print("\nYASA zero-shot: acc %.4f  mF1 %.4f  kappa %.4f" % (res["acc"], res["mf1"], res["kappa"]), flush=True)
print("per-class recall:", {k: round(v,3) for k,v in res["per_class_recall"].items()}, flush=True)
print("predicted Wake %.1f%% vs true %.1f%%" % (100*res["predicted_wake_fraction"], 100*res["true_wake_fraction"]), flush=True)
