"""
run_all.py -- run the full MM-Net revision grid on the fixed 10-fold assignment.
Each config is cached (results/revision/runs/<name>.json); reruns resume, never retrain.
Headline is forced once with save_embed to also write the model, embeddings, predictions.

  KMP_DUPLICATE_LIB_OK=TRUE python revision/run_all.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from mmnet_repro import run_config, summary

CONFIGS = [
    # headline + fusion comparison
    ("headline_concat",  dict(fusion="concat", save_embed=True, force=True)),
    ("attention_cross",  dict(fusion="cross")),
    ("neural_only",      dict(fusion="concat", card_drop=["all"])),
    # leave-one-out (drop one modality, keep the rest)
    ("loo_spo2",         dict(card_drop=["spo2"])),
    ("loo_effort",       dict(card_drop=["effort"])),
    ("loo_pulse_hrv",    dict(card_drop=["pulse_hrv"])),
    ("loo_ecg",          dict(card_drop=["ecg"])),
    ("loo_airflow",      dict(card_drop=["airflow"])),
    ("loo_eog",          dict(eeg_drop=["eog"])),
    ("loo_emg",          dict(eeg_drop=["emg"])),
    # cumulative build-up on the cardiorespiratory stream
    ("cum_spo2",                 dict(card_drop=["pulse_hrv", "ecg", "airflow", "effort"])),
    ("cum_spo2_effort",          dict(card_drop=["pulse_hrv", "ecg", "airflow"])),
    ("cum_spo2_effort_hrv",      dict(card_drop=["ecg", "airflow"])),
    ("cum_spo2_effort_hrv_flow", dict(card_drop=["ecg"])),
]

if __name__ == "__main__":
    t0 = time.time()
    for i, (name, kw) in enumerate(CONFIGS, 1):
        t = time.time()
        r = run_config(name, **kw)
        print(f"[{i:2d}/{len(CONFIGS)}] {summary(r)}  ({time.time()-t:.0f}s)", flush=True)
    print(f"\nall {len(CONFIGS)} configs done in {(time.time()-t0)/60:.1f} min", flush=True)
