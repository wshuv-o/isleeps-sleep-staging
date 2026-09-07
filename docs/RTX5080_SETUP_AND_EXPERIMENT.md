# RTX 5080 machine — setup and the pretrained-EEG experiment

Handoff for running the foundation-model experiment on the Blackwell box. Written to be
followed top to bottom on a machine that has never seen this project.

**Read the two warnings in §0 before installing anything.** They are the only parts of this
document where a wrong move costs real time.

---

## 0. Two things that will bite you

### The current environment will not run on a 5080

The RTX 5080 is Blackwell, compute capability **sm_120**. The project's existing environment
is PyTorch 2.5.0 / CUDA 12.4, which was built for:

```
sm_50, sm_60, sm_61, sm_70, sm_75, sm_80, sm_86, sm_90
```

`sm_120` is absent, so that build fails on Blackwell with
`no kernel image is available for execution on the device`. You need **PyTorch ≥ 2.7 on
CUDA ≥ 12.8**. This is a new environment, not an upgrade of the old one.

### Do not use the 5080 to regenerate published numbers

The paper claims the headline reproduces to four decimal places, and the response letter
says so explicitly. A different CUDA/cuDNN will shift the last digits. That is scientifically
harmless but it breaks a claim we made in writing.

**Rule: the 2060 box owns reproduction. The 5080 box owns new experiments.** If a number is
going into a table that already exists, it comes from the 2060.

---

## 1. Environment

```bash
conda create -n mm5080 python=3.11 -y
conda activate mm5080

# Blackwell needs cu128 wheels; the default index will give you cu124 and fail
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install numpy scipy scikit-learn pandas matplotlib mne openpyxl \
            jupyter nbformat nbclient nbconvert ipykernel
```

**Verify before going further.** This must print `sm_120` in the arch list and complete the
matmul without error:

```python
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0))
print("capability:", "sm_%d%d" % torch.cuda.get_device_capability(0))
print("arch list :", torch.cuda.get_arch_list())
x = torch.randn(4096, 4096, device="cuda")
print("matmul ok :", (x @ x).sum().item() is not None)
```

If `arch list` lacks `sm_120`, the wheel is wrong — you have a cu124 build. Reinstall from
the cu128 index.

On Windows also set `KMP_DUPLICATE_LIB_OK=TRUE`, or MNE and PyTorch will collide over
OpenMP. Every shell block in this document is **bash** (Git Bash on Windows); the
cmd.exe and PowerShell equivalents are given inline where the variable is first set.

---

## 2. Repository and data

```bash
git clone https://github.com/wshuv-o/isleeps-sleep-staging.git
cd isleeps-sleep-staging
```

**The data is not in the repository** — `data/` is git-ignored because it is clinical
recordings. Nothing under `data/` is tracked, so you must bring it across.

| Directory | Size | Needed for |
|---|---|---|
| `data/mm_features/` | 63 MB | **Required.** All existing experiments. 97 npz, one per subject |
| `data/processed7/` | 3.5 GB | **Required for this experiment.** Raw 7-channel signal at 100 Hz — foundation models consume raw signal, not the engineered features |
| `data/multimodal/` | 3.9 GB | Not needed here |
| `data/isruc_mm/`, `data/sleep_edf_mm/` | 15 MB | External validation only |

Copy `data/mm_features/` and `data/processed7/` (≈3.6 GB) by external drive or network share.

Sanity check once copied:

```bash
python -c "
import glob, numpy as np
f = sorted(glob.glob('data/processed7/*.npz'))
print('processed7 subjects:', len(f))           # expect 100
d = np.load(f[0]); print({k: d[k].shape for k in ('x','y')})
print('channels:', list(d['channels']), '| sfreq:', int(d['sfreq']))
"
```

Expect `x (n_epochs, 7, 3000)` at 100 Hz, channels
`C4:M1, C3:M2, O2:M1, O1:M2, E1:M2, E2:M2, EMG`.

---

## 3. Confirm the baseline runs before changing anything

```bash
export KMP_DUPLICATE_LIB_OK=TRUE      # cmd.exe: set KMP_DUPLICATE_LIB_OK=TRUE
                                      # PowerShell: $env:KMP_DUPLICATE_LIB_OK="TRUE"
python -c "
import sys, time; sys.path.insert(0,'MMNet_research/model')
import mmnet_core as C
print('subjects:', len(C.SUBS), '| device:', C.DEV)
t=time.time(); r=C.run_10fold(fusion='concat', seed=42)
print('10-fold in %.1f min' % ((time.time()-t)/60))
print('acc %.4f  AUC %.4f' % (r['acc'][0], r['auc'][0]))
"
```

On the 2060 this takes ~5 minutes and gives `acc 0.7227 / AUC 0.7111`. On the 5080 expect
**1.5–2 minutes**. The numbers should land within about ±0.003 — if they are far off,
something is wrong with the data copy, not with the GPU.

---

## 4. The experiment

### The question

The paper's contribution C2 shows that deep networks built for healthy sleep collapse on
this cohort — 0.61–0.69 accuracy against ~0.85 on healthy sleepers, while 188
physiologically-grounded features hold at 0.72. EEG foundation models are pretrained on
large healthy-sleep corpora, so **the paper predicts they will fail here**.

Nobody has tested that. It is the first thing a reviewer will ask after reading C2.

Both outcomes are publishable, which is why the experiment is worth running:

- **It fails or barely helps** → the strongest form of the finding. Not "small CNNs
  collapse" but "even models pretrained on thousands of healthy nights do not transfer to
  the injured brain, while engineered physiological features do."
- **It works** → you have found what breaks the cohort ceiling, and the paper becomes about
  making foundation models work on pathological sleep.

### Design

Replace the `eeg_enc` MLP (188 engineered features → 128-d) with a pretrained EEG encoder
reading raw signal, and leave everything else identical — cardiorespiratory stream, fusion,
BiLSTM, the validated bypass, both heads. That isolates the encoder as the only variable.

Four arms, three seeds each (42, 1, 7), on the existing ten patient-independent folds:

| Arm | EEG path |
|---|---|
| A | 188 engineered features (published baseline) |
| B | Foundation encoder, **frozen**, linear probe to 128-d |
| C | Foundation encoder, **fine-tuned** end to end |
| D | Foundation encoder architecture, **randomly initialised** — the control |

**Arm D is not optional.** Without it, any gain from B or C could be the architecture rather
than the pretraining, which is exactly the claim being tested.

### Candidate checkpoints

| Model | Params | Notes |
|---|---|---|
| **CBraMod** | ~4 M | Criss-cross transformer, sleep-staging results published, good first choice |
| **LaBraM** | ~5.8 M base | Widely cited, vector-quantised pretraining |
| **BIOT** | ~3 M | Biosignal transformer, handles variable montages — most tolerant of our channel set |
| EEGPT | larger | Try last; heavier |

All are already cited in the paper as [8]–[11]. Start with **CBraMod or BIOT**.

### The adaptation problem — read before coding

This is where the time goes, not the GPU.

1. **Sampling rate.** Our data is 100 Hz; most checkpoints expect 200 Hz or 250 Hz. Resample
   with `scipy.signal.resample_poly`, do not just interpolate.
2. **Channel montage.** Ours is `C4:M1, C3:M2, O2:M1, O1:M2, E1:M2, E2:M2, EMG`. Foundation
   models expect 10-20 names and often a fixed channel order. BIOT is the most forgiving.
   Map C4:M1 → C4, C3:M2 → C3, O2:M1 → O2, O1:M2 → O1 and pass only the four EEG channels
   if the model rejects EOG/EMG.
3. **Epoch length.** We use 30 s = 3000 samples at 100 Hz. Some models want 4 s or 10 s
   windows — pool several sub-windows per epoch and mean-pool the embeddings.
4. **Normalisation.** Ours is per-subject z-scoring of features. Foundation models usually
   expect per-channel µV scaling. Do not double-normalise.

### Feasibility check first — 20 minutes, before committing an afternoon

```python
# load the checkpoint, push one batch of real epochs through, confirm shapes and memory
import numpy as np, torch
d = np.load('data/processed7/SN1.npz')
x = torch.tensor(d['x'][:32]).float().cuda()      # [32, 7, 3000]
print('input', x.shape, 'vram', torch.cuda.memory_allocated()/1e9, 'GB')
# ... model-specific loading here ...
```

If that runs, scale up. If the checkpoint will not load or the montage cannot be mapped,
switch models rather than fighting it.

---

## 5. What to report back

Push results as JSON under `MMNet_research/results/revision/runs/` and commit the notebook
with outputs saved. For each arm, per-fold values, not just means:

```json
{"arm": "B_frozen", "seed": 42,
 "acc": [...10 values...], "mf1": [...], "kappa": [...],
 "auc": [...], "ap": [...]}
```

Then the comparison that matters, on the pooled 30 folds:

- B and C against **A** — does pretraining beat engineered features?
- B and C against **D** — is any gain from *pretraining* or just from the architecture?
- Paired Wilcoxon with Holm correction, and TOST against a margin of one fold SD
  (≈0.034 staging, ≈0.033 AUC) for any claim of no difference.

`MMNet_research/MMNet_Submission/all_codes/notebooks/12_multiseed_ablation.ipynb`
already implements `holm()` and `tost()` — copy them rather than rewriting. Note that all
experiment notebooks live under `MMNet_research/MMNet_Submission/all_codes/notebooks/`,
**not** under `MMNet_research/notebooks/`, which holds older working copies.

---

## 6. House rules for this project

- **All training runs inside a notebook**, executed with outputs saved. Not scripts with
  pasted results.
  ```bash
  cd MMNet_research/MMNet_Submission/all_codes/notebooks
  python -m nbconvert --to notebook --execute --inplace \
         --ExecutePreprocessor.timeout=36000 <nb>.ipynb
  ```
- Checkpoint after every run so a killed session does not lose hours. See how
  `MMNet_research/MMNet_Submission/all_codes/notebooks/12_multiseed_ablation.ipynb`
  writes its JSON after each condition — two sessions were killed mid-sweep during this
  project and lost nothing because of it.
- Report negative results as they come. Three findings already in this paper came out
  against us and are stated in the manuscript.
- Commits on this repo are authored as `EsmeAbha <esmechowdhuryabha@gmail.com>`.

---

## 7. If you get stuck

The fastest diagnostic is whether `mmnet_core` runs at all on the new box (§3). If that
reproduces, the environment is fine and the problem is in the foundation-model adaptation —
almost always montage or sampling rate. If §3 fails, the problem is the environment or the
data copy, and the foundation model is a distraction until it is fixed.
