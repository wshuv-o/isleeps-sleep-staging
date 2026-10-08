"""Rebuild every cache the MM-Net runs read, from the public iSLEEPS release.

    python MMNet_research/preprocessing/rebuild_data.py            # everything
    python MMNet_research/preprocessing/rebuild_data.py --from features

Stages, in order (--from starts part-way; finished files are skipped, so a
re-run after an interruption resumes):

  download     Figshare 29253068 (97 EDF + 100 annotation workbooks +
               subject_description.xlsx, 17.5 GB) -> data/Dataset/
               Zenodo 14873844 SN2, SN13, SN17 (absent from Figshare) -> data/zenodo/
  processed7   build_npz_full.py          EDF -> 7-channel 30 s epochs at 100 Hz
  multimodal   build_multimodal.py        + 7 cardiorespiratory channels at 25 Hz
               rebuild_missing_subjects.py   the same for SN2, SN13, SN17
  features     extract_mm_features.py     188 neural + 14 cardio descriptors
               rebuild_missing_subjects.py features   the same for SN2, SN13, SN17
  labram       build_labram_cache.py      frozen LaBraM embedding (weights from the
               braindecode/labram-pretrained checkpoint on Hugging Face)

Checked against the caches the published results were trained on, for SN2:
processed7 and multimodal are bit-identical, the 14 cardio features are identical,
and of the 188 neural features only the EOG spectral edge (spec_edge_c4) moves by
one frequency bin on a few epochs, from float rounding.

Needs mne, pyedflib, braindecode, torch, openpyxl, scipy (see requirements.txt).
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MM = os.path.dirname(HERE)
REPO = os.path.dirname(MM)
DATA = os.path.join(REPO, "data")
FIGSHARE = "https://api.figshare.com/v2/articles/29253068/files?page_size=1000"
ZENODO = "https://zenodo.org/api/records/14873844/files/%s/content"
ZENODO_ONLY = ("SN2", "SN13", "SN17")
STAGES = ("download", "processed7", "multimodal", "features", "labram")


def _get(url, dst, size=None):
    if os.path.exists(dst) and (size is None or os.path.getsize(dst) == size):
        return False
    tmp = dst + ".part"
    urllib.request.urlretrieve(url, tmp)
    if size is not None and os.path.getsize(tmp) != size:
        raise IOError("%s: got %d bytes, expected %d" % (dst, os.path.getsize(tmp), size))
    os.replace(tmp, dst)
    return True


def download():
    ds = os.path.join(DATA, "Dataset")
    os.makedirs(ds, exist_ok=True)
    files = json.load(urllib.request.urlopen(FIGSHARE))
    want = [f for f in files if f["name"] == "subject_description.xlsx"
            or (f["name"].startswith("SN") and f["name"].endswith((".edf", ".xlsx")))]
    print("Figshare: %d files, %.1f GB" % (len(want), sum(f["size"] for f in want) / 1e9), flush=True)
    for i, f in enumerate(sorted(want, key=lambda f: f["size"])):
        if _get(f["download_url"], os.path.join(ds, f["name"]), f["size"]):
            print("  [%d/%d] %s" % (i + 1, len(want), f["name"]), flush=True)
    zd = os.path.join(DATA, "zenodo")
    os.makedirs(zd, exist_ok=True)
    for sid in ZENODO_ONLY:
        for ext in (".edf", ".xlsx"):
            if _get(ZENODO % (sid + ext), os.path.join(zd, sid + ext)):
                print("  zenodo %s%s" % (sid, ext), flush=True)
    full = os.path.join(DATA, "full100")
    os.makedirs(full, exist_ok=True)
    src = os.path.join(ds, "subject_description.xlsx")
    dst = os.path.join(full, "subject_description.xlsx")
    if not os.path.exists(dst):
        with open(src, "rb") as a, open(dst, "wb") as b:
            b.write(a.read())


def run(script, *args):
    print("\n== %s %s" % (os.path.relpath(script, REPO), " ".join(args)), flush=True)
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
    subprocess.run([sys.executable, script] + list(args), check=True, cwd=REPO, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=STAGES, default="download")
    a = ap.parse_args()
    todo = STAGES[STAGES.index(a.start):]
    if "download" in todo:
        download()
    if "processed7" in todo:
        run(os.path.join(HERE, "build_npz_full.py"))
    if "multimodal" in todo:
        run(os.path.join(HERE, "build_multimodal.py"))
        run(os.path.join(HERE, "rebuild_missing_subjects.py"), "multimodal")
    if "features" in todo:
        run(os.path.join(HERE, "extract_mm_features.py"))
        run(os.path.join(HERE, "rebuild_missing_subjects.py"), "features")
    if "labram" in todo:
        run(os.path.join(MM, "foundation", "build_labram_cache.py"), "--variant", "labram")
    print("\ndone. MM-Net caches are under %s" % DATA)


if __name__ == "__main__":
    main()
