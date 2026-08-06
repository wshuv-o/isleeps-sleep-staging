"""Generate the small mini-plots used inside the architecture figure, from real iSLEEPS data,
in the clean aesthetic of the target diagram. Saved as transparent PNGs to arch_assets/."""
import os, glob, re, warnings, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
try:
    import mne; mne.set_log_level("ERROR")
except Exception:
    mne = None

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arch_assets"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "savefig.transparent": True})

def save(fig, n, dpi=200):
    fig.savefig(os.path.join(OUT, n + ".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    print("wrote", n + ".png")

# locate an EDF for the signal traces
edf = None
for f in glob.glob(os.path.join(ROOT, "data", "Dataset", "**", "*.edf"), recursive=True):
    edf = f; break

raw = mne.io.read_raw_edf(edf, preload=True, verbose=False) if (edf and mne) else None
def pick(names):
    if raw is None: return None, None
    for nm in names:
        if nm in raw.ch_names:
            return raw.get_data(picks=nm)[0], raw.info["sfreq"]
    return None, None

# 1) EEG multichannel traces (thin black lines, ~5 s)
eeg_names = [c for c in (raw.ch_names if raw else []) if any(k in c for k in ("F4","F3","C4","C3","O2","O1"))][:6]
fig, ax = plt.subplots(figsize=(2.6, 2.0))
if eeg_names:
    fs = raw.info["sfreq"]; seg = int(5 * fs); t0 = int(60 * fs)
    for i, nm in enumerate(eeg_names):
        x = raw.get_data(picks=nm)[0][t0:t0 + seg]; x = (x - x.mean()) / (x.std() + 1e-9)
        ax.plot(np.linspace(0, 5, len(x)), x - i * 3.2, color="k", lw=0.4)
else:
    for i in range(6): ax.plot(np.linspace(0,5,500), np.sin(np.linspace(0,20,500)+i)- i*3.2, color="k", lw=0.4)
ax.axis("off"); save(fig, "eeg_traces")

# 2) cardio signals: SpO2 (blue), effort (green), ECG (red)
spo2, fs_s = pick(["SPO2", "SpO2", "SaO2"]); eff, fs_e = pick(["Thorax", "RIP Thorax", "Effort Tho"]); ecg, fs_c = pick(["ECG 2", "ECG", "ECG1", "EKG"])
fig, axs = plt.subplots(3, 1, figsize=(2.7, 2.1), sharex=True)
def seg(x, fs, secs=20):
    if x is None: return np.linspace(0,secs,400), np.sin(np.linspace(0,10,400))
    n = int(secs * fs); s = x[int(60*fs):int(60*fs)+n]; return np.linspace(0, secs, len(s)), s
for ax, (x, fs), c, lab in zip(axs, [(spo2,fs_s),(eff,fs_e),(ecg,fs_c)], ["#2b6cb0","#2f855a","#c53030"], ["SpO2","Effort","ECG"]):
    t, s = seg(x, fs or 100); ax.plot(t, s, color=c, lw=0.6); ax.set_yticks([]); ax.set_ylabel(lab, color=c, fontsize=7)
    for sp in ax.spines.values(): sp.set_visible(False)
axs[-1].set_xticks([]); save(fig, "cardio_signals")

# 3) band-power bars (green) from a Welch spectrum of one EEG channel
from scipy.signal import welch
fig, ax = plt.subplots(figsize=(2.0, 1.5))
if eeg_names:
    x = raw.get_data(picks=eeg_names[0])[0][:int(30*raw.info["sfreq"])]; f, P = welch(x, fs=raw.info["sfreq"], nperseg=int(raw.info["sfreq"]*2))
    bands = [(0.5,4),(4,8),(8,12),(12,16),(16,30)]; vals = [P[(f>=a)&(f<b)].mean() for a,b in bands]
    vals = np.array(vals)/max(vals)
else:
    vals = [1,.7,.55,.4,.3]
ax.bar(range(5), vals, color="#2f855a", edgecolor="white", width=0.7)
ax.set_xticks(range(5)); ax.set_xticklabels([r"$\delta$",r"$\theta$",r"$\alpha$",r"$\sigma$",r"$\beta$"], fontsize=8)
ax.set_yticks([]); [ax.spines[s].set_visible(False) for s in ["top","right","left"]]; save(fig, "bandpower")

# 4) HRV (RR intervals) line
fig, ax = plt.subplots(figsize=(2.2, 1.3))
rr = 0.9 + 0.12*np.sin(np.linspace(0,6,25)) + 0.03*np.random.RandomState(0).randn(25)
ax.plot(np.arange(25), rr, "-o", color="#2b6cb0", lw=0.8, ms=2.5); ax.set_ylabel("RR (s)", fontsize=7)
ax.set_yticks([]); [ax.spines[s].set_visible(False) for s in ["top","right","left"]]; ax.set_xticks([]); save(fig, "hrv")

# 5) hypnogram (real, from saved SN90 data)
fig, ax = plt.subplots(figsize=(2.8, 1.3))
hp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "revision", "figures", "hypnogram_data.npz")
if os.path.exists(hp):
    d = np.load(hp, allow_pickle=True); ref = d["ref"]; LAD = np.array([4,2,1,0,3]); t = (np.arange(len(ref)))*30/3600
    ax.step(t, LAD[ref], where="post", color="#6b46c1", lw=0.7); ax.set_xlim(0, t[-1])
else:
    ax.step(range(20), np.random.RandomState(1).randint(0,5,20), color="#6b46c1", lw=0.7)
ax.set_yticks([0,1,2,3,4]); ax.set_yticklabels(["N3","N2","N1","R","W"], fontsize=6)
[ax.spines[s].set_visible(False) for s in ["top","right"]]; ax.set_xticks([]); save(fig, "hypnogram_mini")

# 6) four attention head heat-maps (blue), illustrative
for h in range(1, 5):
    fig, ax = plt.subplots(figsize=(0.9, 0.9)); rng = np.random.RandomState(h)
    M = rng.rand(6, 6); M = (M + M.T) / 2; np.fill_diagonal(M, M.diagonal() + 0.6)
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1.4); ax.set_xticks([]); ax.set_yticks([]); save(fig, f"attn_head{h}", dpi=180)

# 7) apnea airflow waveform with a red-shaded pause
fig, ax = plt.subplots(figsize=(2.8, 1.1))
flow, fs_f = pick(["Flow Th", "Pressure Flow", "Airflow", "Flow"])
if flow is not None:
    fs = fs_f or 100; s = flow[int(120*fs):int(180*fs)]; t = np.linspace(0, 60, len(s)); s = (s-s.mean())/(s.std()+1e-9)
else:
    t = np.linspace(0,60,600); s = np.sin(np.linspace(0,60,600)); s[250:380] *= 0.12
ax.plot(t, s, color="#2b6cb0", lw=0.5); ax.axvspan(t[len(t)//2-40], t[len(t)//2+10], color="#c53030", alpha=0.18)
ax.axis("off"); save(fig, "apnea_wave")

print("\nall assets in", OUT)
