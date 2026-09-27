"""Items 8b and 12: which airflow sensors iSLEEPS provides, and which recordings
exist as EDF at all.

Reads only the 256-byte EDF headers and label blocks of data/Dataset/Batch-*/SN*.edf
(no signal data), and lists the recordings of the 99-patient cohort with no EDF on
disk. The cardiorespiratory channel map MM-Net uses is preprocessing/build_multimodal.py
(CARD: ECG, 'Flow Th', Thorax, Abdomen, Effort, SpO2, Pulse).

  python edf_channels.py
"""
import collections
import glob
import os

import _common as C


def labels(path):
    with open(path, "rb") as f:
        h = f.read(256)
        ns = int(h[252:256])
        lab = f.read(ns * 16)
    return [lab[i * 16:(i + 1) * 16].decode("latin1").strip() for i in range(ns)]


files = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(C.DATA, "Dataset", "Batch-*", "SN*.edf"))}
cnt = collections.Counter()
for s, p in files.items():
    cnt.update(labels(p))
C.save("edf_channels.json", dict(
    n_edf=len(files), label_counts=dict(cnt),
    cohort_without_edf_on_disk=[s for s in C.SUBJECTS if s not in files],
    airflow_sensors_present={"Flow Th (thermal, oronasal)": cnt.get("Flow Th", 0),
                             "Pressure Flow (nasal pressure)": cnt.get("Pressure Flow", 0)},
    mmnet_reads="Flow Th only (preprocessing/build_multimodal.py CARD 'Flow' <- ['Flow Th'])",
    huttunen_port_reads="SpO2, Flow Th, Sum Effort, C4:M1 (run_huttunen.py); nasal pressure omitted"))
print(len(files), cnt["Flow Th"], cnt["Pressure Flow"], [s for s in C.SUBJECTS if s not in files])
