"""Remap the modality-ablation masks for the final model's inputs.

The ablation grid is the evidence for C4 (per-modality attribution). Its masks
were written for the manuscript's inputs and are WRONG for the final model in two
different ways -- neither raises an error, both silently produce a plausible
number, so both have to be fixed before any C4 figure is regenerated.

1. CARDIO. CARD_GROUPS indexes the 14 engineered features:

       spo2 [0,1,2,3]  pulse_hrv [4,5]  ecg [6,7]  airflow [8,9]  effort [10..13]

   The final model feeds the RAW 7-channel signal flattened to 5250, where
   channel c occupies [c*750:(c+1)*750] in the order
   ECG, Flow, Thorax, Abdomen, Effort, SpO2, Pulse. So card_drop='spo2' would
   currently zero four arbitrary samples near the start of the ECG trace. The
   remap zeroes whole channels, which is the honest analogue of removing a
   modality.

2. EEG. EEGM masks are 188-long, matching the engineered feature block. The
   final model's neural input is 388-d: the 188 features followed by a 200-d
   LaBraM embedding. Left unpatched, eeg_drop='eeg' would zero the engineered
   EEG features while the LaBraM embedding -- built from those same four EEG
   derivations -- silently carried the information through, and the ablation
   would understate how much EEG matters.

   LaBraM sees ONLY the four EEG channels (EOG and EMG have no canonical 10-20
   name and are dropped at cache time), so the 'eog' and 'emg' masks stay
   correct as they are; only 'eeg' has to be widened to cover the embedding.
"""
import numpy as np

N_CH, N_T = 7, 750
RAW_CHANNELS = ["ECG", "Flow", "Thorax", "Abdomen", "Effort", "SpO2", "Pulse"]

# manuscript modality name -> the raw channels it corresponds to
RAW_GROUPS = {
    "spo2":      ["SpO2"],
    "pulse_hrv": ["Pulse"],
    "ecg":       ["ECG"],
    "airflow":   ["Flow"],
    # the manuscript's "effort" covers both belts and the derived sum
    "effort":    ["Thorax", "Abdomen", "Effort"],
}


def raw_card_groups():
    """{modality: flat indices into the 5250-d raw cardio vector}."""
    out = {}
    for name, chans in RAW_GROUPS.items():
        idx = []
        for c in chans:
            k = RAW_CHANNELS.index(c)
            idx.extend(range(k * N_T, (k + 1) * N_T))
        out[name] = idx
    out["all"] = list(range(N_CH * N_T))
    return out


def widened_eeg_masks(C, n_eeg_total, n_features=188):
    """EEG masks widened to the concatenated neural input.

    The embedding block is appended to the 'eeg' mask because LaBraM is built
    from the EEG derivations alone; 'eog' and 'emg' keep their original extent.
    """
    out = {}
    for name, mask in C.EEGM.items():
        wide = np.zeros(n_eeg_total, bool)
        wide[:n_features] = mask
        if name == "eeg":
            wide[n_features:] = True        # the LaBraM embedding is EEG-derived
        out[name] = wide
    return out


def apply(C, n_eeg_total, raw_cardio=True, n_features=188):
    """Patch C in place; returns a restore callable."""
    old_card, old_eegm = C.CARD_GROUPS, C.EEGM
    if raw_cardio:
        C.CARD_GROUPS = raw_card_groups()
    C.EEGM = widened_eeg_masks(C, n_eeg_total, n_features)
    sizes = {k: len(v) for k, v in C.CARD_GROUPS.items()}
    print("cardio masks -> %s" % sizes, flush=True)
    print("eeg masks    -> %s" % {k: int(v.sum()) for k, v in C.EEGM.items()}, flush=True)

    def restore():
        C.CARD_GROUPS, C.EEGM = old_card, old_eegm
    return restore
