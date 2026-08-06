"""
parse_subject_description.py  —  TODO-0 tool

Reads the iSLEEPS clinical-metadata file `subject_description.xlsx` (63 columns,
100 subjects) and reports, per column: dtype, % populated (missingness), and a
few example values. Then flags the Pillar-(ii) fields the study depends on
(lesion side/hemisphere, lesion location, stroke severity NIHSS/mRS, time from
onset) so you can see at a glance how strong the lesion-aware analysis can be.

STATUS: `subject_description.xlsx` is NOT in the Zenodo 40-subject release.
It ships only via iHUB-Data (all 100 subjects, free registration):
  https://india-data.org/dataset-details/0b801dfa-4e42-4ec6-9c56-c6892b907ed2
Download it to data/ and run:
  python metadata/parse_subject_description.py --xlsx data/subject_description.xlsx
"""
import argparse
import os
import re

import pandas as pd

# Regexes for the Pillar-(ii) fields we most want to confirm exist + are populated.
PILLAR2_PATTERNS = {
    "hemisphere / lesion side": r"hemisph|side|left|right|lateral|ipsi|contra",
    "lesion location":          r"lesion|territory|cortic|subcortic|infarct|location|region|mca|aca|pca",
    "stroke severity":          r"nihss|mrs|rankin|severity|barthel",
    "time from onset":          r"onset|days?\s*(since|from|post)|duration|time.*stroke|stroke.*time",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="data/subject_description.xlsx")
    ap.add_argument("--sheet", default=0, help="sheet name or index")
    args = ap.parse_args()

    if not os.path.exists(args.xlsx):
        raise SystemExit(
            f"Not found: {args.xlsx}\n"
            "subject_description.xlsx is iHUB-Data-only (not on Zenodo). "
            "Register, download it into data/, then re-run."
        )

    df = pd.read_excel(args.xlsx, sheet_name=args.sheet)
    n = len(df)
    print(f"{args.xlsx}: {n} subjects x {df.shape[1]} columns\n")

    print(f"{'#':>3} {'column':40s} {'dtype':10s} {'%filled':>8s}  example")
    for i, col in enumerate(df.columns):
        s = df[col]
        filled = 100 * s.notna().mean()
        ex = next((repr(v) for v in s.dropna().tolist()[:1]), "")
        print(f"{i:3d} {str(col)[:40]:40s} {str(s.dtype):10s} {filled:7.1f}%  {ex[:32]}")

    print("\n=== Pillar (ii) field check ===")
    cols_lc = {c: str(c).lower() for c in df.columns}
    for label, pat in PILLAR2_PATTERNS.items():
        hits = [c for c, lc in cols_lc.items() if re.search(pat, lc)]
        if hits:
            for c in hits:
                filled = 100 * df[c].notna().mean()
                print(f"  [FOUND] {label:24s} -> '{c}'  ({filled:.0f}% populated)")
        else:
            print(f"  [missing] {label}")


if __name__ == "__main__":
    main()
