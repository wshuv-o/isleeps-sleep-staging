"""Download a Sleep-EDF Expanded (sleep-cassette) subset from PhysioNet.

Needed as the CONTROL for the zero-shot experiment. A healthy-trained stager
scoring 0.308 on iSLEEPS is only evidence of a domain gap if the same checkpoint,
through the same preprocessing, scores properly on healthy sleep. Without that,
"the injured brain breaks the model" and "our input pipeline is wrong" produce
identical numbers.

sleep-cassette is the standard healthy benchmark (subjects 25-101, no medication).
Each recording is a PSG .edf plus a Hypnogram .edf of expert annotations.
"""
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
OUT = os.path.join(os.getcwd(), "data", "sleep_edf_raw")
N_REC = int(os.environ.get("SEDF_N", "40"))
WORKERS = 6


def listing():
    with urllib.request.urlopen(BASE, timeout=60) as r:
        html = r.read().decode("utf-8", "ignore")
    psg = sorted(set(re.findall(r"(SC4\d{3}[A-Z0-9]*-PSG\.edf)", html)))
    hyp = sorted(set(re.findall(r"(SC4\d{3}[A-Z0-9]*-Hypnogram\.edf)", html)))
    # pair by the 6-character subject/night stem, e.g. SC4001
    hmap = {h[:6]: h for h in hyp}
    pairs = [(p, hmap[p[:6]]) for p in psg if p[:6] in hmap]
    return pairs


def get(name):
    dst = os.path.join(OUT, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return name, os.path.getsize(dst), True
    tmp = dst + ".part"
    with urllib.request.urlopen(BASE + name, timeout=300) as r, open(tmp, "wb") as fh:
        while True:
            c = r.read(1 << 20)
            if not c:
                break
            fh.write(c)
    os.replace(tmp, dst)
    return name, os.path.getsize(dst), False


def main():
    os.makedirs(OUT, exist_ok=True)
    pairs = listing()[:N_REC]
    files = [f for pr in pairs for f in pr]
    print("recordings: %d  (%d files)" % (len(pairs), len(files)), flush=True)
    t0 = time.time(); total = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(get, f): f for f in files}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                name, size, cached = fu.result()
                total += size
                print("[%s] %-34s %7.1f MB  (%d/%d, %.1f min)"
                      % ("skip" if cached else "ok", name, size / 1e6, i, len(files),
                         (time.time() - t0) / 60), flush=True)
            except Exception as e:
                print("[err] %s: %s" % (futs[fu], e), flush=True)
    print("\n=== %d files, %.2f GB in %.1f min ==="
          % (len(files), total / 1e9, (time.time() - t0) / 60))


if __name__ == "__main__":
    sys.exit(main())
