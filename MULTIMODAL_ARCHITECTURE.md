# Multimodal multi-task sleep network — complete architecture specification

Source of truth: [model/mm_feature_net.py](model/mm_feature_net.py). Every shape and
parameter count below is emitted by the code, not hand-derived. Total **856,326 parameters**
(headline `cross` fusion). Notation: `B` = batch (nights), `L` = epochs per window (20),
`F` = feature dim. A "token" is one 30-second epoch.

---

## 0. Design principles (why it is shaped this way)

1. **Features, not raw signal.** On 96 patients a from-scratch CNN cannot re-learn what 188
   engineered EEG features already encode (it plateaus at 0.65). So the two streams consume
   *features*, and the network spends its capacity on the two things deep nets do better than
   feature pipelines: the **cross-modal interaction** and the **overnight temporal grammar**.
2. **Two modalities, two tasks, one model.** EEG features drive staging; cardiorespiratory
   features drive respiratory-event detection. Multi-task training keeps the cardio stream
   load-bearing by construction (it has its own supervised head).
3. **Every block is ablatable.** Fusion has three modes (`cross` / `concat` / `eeg_only`) and
   the respiratory head has a switchable direct pathway, so each component's contribution is
   *measured*, not assumed.
4. **Sized for the regime.** Compact (0.86 M params, 77% of it the BiLSTM), LayerNorm not
   BatchNorm (short windows, patient-mixed batches), per-patient feature standardization.

---

## 1. Inputs

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `feeg` | `[B, L, 188]` | float32 | per-epoch EEG/EOG/EMG features, per-patient z-scored |
| `fcard` | `[B, L, 14]` | float32 | per-epoch cardiorespiratory features, per-patient z-scored |
| `y_stage` | `[B, L]` | int64 | AASM stage label 0..4 (W,N1,N2,N3,R) |
| `y_apnea` | `[B, L]` | float32 | per-epoch respiratory-event label {0,1} |

**EEG feature block (188):** per channel (7) — absolute + relative band power (δ θ α σ β),
spectral entropy, spectral-edge freq, 3 Hjorth descriptors, time-domain stats; plus event
features — spindle density + Hilbert-envelope moments (sigma band), slow-wave amplitudes,
EOG movement energy, tonic EMG level.

**Cardiorespiratory feature block (14):** SpO₂ mean/min/std/desaturation-depth (idx 0–3),
pulse mean/std (4–5), ECG std/line-length (6–7), **airflow std/line-length (8–9)**,
thoracic/abdominal/summed-effort std (10–12), thoraco-abdominal asynchrony (13).
Indices 8–9 are the airflow features deleted in the `cross_noflow` non-circularity check.

---

## 2. Forward pass (verified shape trace, B=2, L=20)

```
feeg  [2,20,188] ──reshape──> [40,188] ──EEG encoder──────> e  [40,128]
fcard [2,20, 14] ──reshape──> [40, 14] ──Cardio encoder───> c  [40, 64]
                                          c ──card_proj────> cp [40,128]
(e, cp) ─────────────── Cross-modal fusion ───────────────> fz [40,128]
fz ──reshape──> [2,20,128] ──BiLSTM (2 layers, bidir)─────> h  [2,20,256]
h ─────────────────────── Staging head ───────────────────> stage [2,20,5]
[h ‖ c] [2,20,320] ────── Respiratory head ───────────────> apnea [2,20,1] -> squeeze [2,20]
```

The two streams run on the flattened `B*L` epoch axis (per-epoch, order-independent); the
BiLSTM is the only block that sees the epoch sequence.

---

## 3. Module-by-module detail

### 3.1 EEG feature encoder — `FeatMLP(188 → 128)` · 41,216 params
```
Linear(188, 128)  ->  LayerNorm(128)  ->  GELU  ->  Dropout(0.3)
Linear(128, 128)  ->  LayerNorm(128)  ->  GELU  ->  Dropout(0.3)
```
Input `[B*L,188]` → output `e [B*L,128]`. Two-layer MLP that lifts the sparse, heterogeneous
188-vector into a dense 128-d embedding. LayerNorm stabilizes across the wide dynamic range of
band-power vs event features.

### 3.2 Cardiorespiratory feature encoder — `FeatMLP(14 → 64)` · 5,376 params
```
Linear(14, 64)  ->  LayerNorm(64)  ->  GELU  ->  Dropout(0.3)
Linear(64, 64)  ->  LayerNorm(64)  ->  GELU  ->  Dropout(0.3)
```
Input `[B*L,14]` → output `c [B*L,64]`. Deliberately smaller than the EEG stream: 14 slow,
quasi-periodic descriptors carry far less structure than 188 EEG features, so a wider net would
only overfit. `c` is used **twice** — once into fusion, once directly into the respiratory head.

### 3.3 Cardio projection — `Linear(64 → 128)` · 8,320 params  *(cross fusion only)*
Lifts `c` to the shared 128-d width so it can co-attend with `e`. → `cp [B*L,128]`.

### 3.4 Cross-modal fusion — `CrossFusion(d=128, heads=4)` · 99,456 params
The novel core. Treat the epoch's EEG and cardio embeddings as a length-2 token sequence and
let them attend to each other.
```
tok = stack([e, cp], dim=1) + mtype           # [B*L, 2, 128]   mtype: learned modality-type emb (2,128)
a   = MultiheadAttention(128, heads=4)(tok,tok,tok)   # self-attention over the 2 tokens
a   = LayerNorm(128)(tok + a)                 # residual + norm
fz  = Linear(256,128)(a.reshape(B*L, 2*128))  # -> GELU -> Dropout   fuse the 2 refined tokens
```
Output `fz [B*L,128]`. Components: modality-type embeddings (256), MHA in/out projections
(66,048), LayerNorm (256), feed-forward `Linear(256,128)` (32,896).
Rationale: in a severe-apnea cohort the respiratory context could *reshape* the EEG
representation, not merely sit beside it. **Measured result: attention ties plain
concatenation** — reported honestly; concatenation is the default.

### 3.5 Temporal decoder — `BiLSTM(128 → 128, 2 layers, bidirectional)` · 659,456 params
```
LSTM(input=128, hidden=128, num_layers=2, bidirectional=True, dropout=0.3)
```
Input `fz.reshape(B,L,128)` → output `h [B,L,256]` (256 = 2×128 forward‖backward). This is the
capacity center of the model (77% of params). It learns the overnight stage grammar that a
classical pipeline imposes with a separate HMM — a whole window of 20 epochs (10 min) of
bidirectional context per prediction.

### 3.6 Staging head — `Linear(256 → 5)` · 1,285 params
`h [B,L,256]` → `stage_logits [B,L,5]`. Softmax over the 5 AASM classes at inference; HMM Viterbi
smoothing applied post-hoc (Section 5).

### 3.7 Respiratory head — MLP `320 → 128 → 1` · 41,217 params  *(direct cardio pathway)*
```
apnea_in = cat([h, c_seq], dim=-1)            # [B,L, 256+64 = 320]   c_seq = c reshaped (zeros if eeg_only)
Linear(320,128) -> GELU -> Dropout(0.3) -> Linear(128,1) -> squeeze   # [B,L]
```
The head reads the recurrent context **and a direct copy of the per-epoch cardio embedding
`c`**. This bypass is the key fix: without it the respiratory decision must survive a fusion
trained mostly for staging, and SpO₂-desaturation signal is attenuated. Adding it lifted apnea
AUC 0.66→0.71. For the `eeg_only` ablation, `c_seq` is zeroed so the head is truly
cardio-free.

---

## 4. Fusion variants (the ablation switch)

| mode | fusion block | cardio into apnea head | params |
|---|---|---|---|
| `cross` | CrossFusion(128) + card_proj | yes (`c`) | 856,326 |
| `concat` | `Linear(128+64,128)->GELU->Drop` | yes (`c`) | ~773 K |
| `eeg_only` | `Linear(128,128)->GELU->Drop` | **no** (zeros) | ~765 K |

`cross_noflow` = `cross` with airflow feature columns 8,9 zeroed at the input (non-circularity
test); same parameter count as `cross`.

---

## 5. Loss, training, inference

**Joint loss** (both tasks weighted equally):
```
L = CE_stage(sqrt-inverse-freq class weights)  +  1.0 * BCE_apnea(pos_weight = neg/pos)
```
- Staging CE uses **√(inverse frequency)** weighting — full inverse-freq over-corrects rare
  N1/N3 and costs accuracy; √ is the acc/macro-F1 sweet spot.
- Apnea BCE uses `pos_weight = #neg/#pos` (~5.3) so the 16%-positive head does not collapse.

**Optimization:** AdamW (lr 1e-3, wd 1e-4), cosine annealing, grad-clip 5.0, batch 32,
windows of L=20 with stride L/2, early stop on validation staging accuracy.

**Inference — staging:** per-epoch posteriors → **HMM Viterbi** with transition matrix + prior
counted from training labels (+~1pp, removes single-epoch flips). Respiratory head left
unsmoothed.

---

## 6. Parameter budget

| module | params | share |
|---|---:|---:|
| BiLSTM temporal decoder | 659,456 | 77.0% |
| Cross-modal fusion | 99,456 | 11.6% |
| EEG feature encoder | 41,216 | 4.8% |
| Respiratory head | 41,217 | 4.8% |
| card_proj | 8,320 | 1.0% |
| Cardio feature encoder | 5,376 | 0.6% |
| Staging head | 1,285 | 0.2% |
| **total** | **856,326** | 100% |

---

## 7. Component → task impact map (what the ablation proved)

| component | staging effect | apnea effect |
|---|---|---|
| Cardio stream (vs eeg_only) | none (0.721 → 0.721) | **+4.5pp AUC** (0.655 → 0.700) |
| Cross-modal attention (vs concat) | ties (0.715 vs 0.721) | ties (0.698 vs 0.700) |
| Direct cardio pathway (apnea head) | n/a | **+~2pp AUC** (0.695 → 0.71) |
| √ class weighting | **+2pp acc** | n/a |
| HMM smoothing | **+1pp acc** | n/a (unsmoothed) |
| Airflow deletion (cross_noflow) | none | none (0.707) → **non-circular** |
