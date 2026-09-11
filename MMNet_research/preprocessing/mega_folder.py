"""Minimal read-only client for a public MEGA folder.

Why this exists, and why it is in the repository rather than a scratch
directory: ISRUC-Sleep is cited from its official archive, but that archive was
unreachable when this work needed it (see DATA_NOTES.md), and the corpus was
obtained from a public MEGA mirror instead. MEGA encrypts client-side --- the
folder key travels in the URL fragment and never reaches their server --- so no
ordinary downloader can fetch it, and a reader trying to reproduce
`data/isruc/` would otherwise have no route at all.

Scope is deliberately small: list a public folder and download from it. No
login, no upload, no private nodes.

Two things about MEGA's share format that are easy to get wrong, both of which
produce silent garbage rather than an error:

  * Node keys in a share are not always a multiple of the AES block size. The
    ragged tail has to be dropped before ECB decryption or the cipher raises,
    and padding it instead yields a key that decrypts to noise.
  * A node's `k` field can carry several `handle:key` pairs joined by `/`,
    because a node may be shared by more than one route. Which pair applies
    depends on how the share was built, so every candidate is tried and the one
    whose attribute blob starts with the `MEGA{` marker is kept. Taking the
    first pair works often enough to look correct and fails on some folders.

  python mega_folder.py --url "https://mega.nz/folder/<id>#<key>" --list
  python mega_folder.py --url "..." --out data/isruc
"""
import argparse
import base64
import json
import os
import struct
import sys
import time

import requests
from Crypto.Cipher import AES
from Crypto.Util import Counter

API = "https://g.api.mega.co.nz/cs"


def b64d(s):
    """MEGA's URL-safe base64, unpadded."""
    s = s.replace("-", "+").replace("_", "/")
    return base64.b64decode(s + "=" * ((4 - len(s) % 4) % 4))


def to_words(b):
    return struct.unpack(">%dI" % (len(b) // 4), b[:len(b) // 4 * 4])


def from_words(w):
    return struct.pack(">%dI" % len(w), *w)


def decrypt_key(key_bytes, master):
    """AES-ECB over 16-byte blocks; a ragged tail is dropped, not padded."""
    usable = len(key_bytes) // 16 * 16
    if usable == 0:
        return None
    c = AES.new(master, AES.MODE_ECB)
    return c.decrypt(key_bytes[:usable])


def decrypt_attr(attr_bytes, key16):
    """Attributes are AES-CBC with a zero IV; valid ones start with 'MEGA{'."""
    if len(attr_bytes) % 16:
        attr_bytes = attr_bytes[:len(attr_bytes) // 16 * 16]
    if not attr_bytes:
        return None
    try:
        raw = AES.new(key16, AES.MODE_CBC, b"\0" * 16).decrypt(attr_bytes)
    except ValueError:
        return None
    raw = raw.rstrip(b"\0")
    if not raw.startswith(b"MEGA{"):
        return None
    try:
        return json.loads(raw[4:].decode("utf-8", "ignore"))
    except ValueError:
        return None


def node_keys(node, folder_key):
    """Every plausible file key for a node, best first.

    The `k` field is `handle:key` pairs joined by '/'. More than one can be
    present; only one decrypts to a readable attribute blob.
    """
    out = []
    for part in node.get("k", "").split("/"):
        if ":" not in part:
            continue
        enc = decrypt_key(b64d(part.split(":", 1)[1]), folder_key)
        if enc:
            out.append(enc)
    return out


def unpack_file_key(k):
    """A 32-byte file key folds down to the 16-byte AES key, plus IV and MAC."""
    w = to_words(k)
    if len(w) < 8:
        return None, None
    key = from_words((w[0] ^ w[4], w[1] ^ w[5], w[2] ^ w[6], w[3] ^ w[7]))
    iv = w[4:6]
    return key, iv


def api(payload, node_id):
    r = requests.post(API, params={"n": node_id}, data=json.dumps([payload]), timeout=60)
    r.raise_for_status()
    res = r.json()
    if isinstance(res, int):
        raise RuntimeError("MEGA API error %d" % res)
    return res[0]


def list_folder(url):
    """-> (node_id, folder_key, [nodes]) with names and parents resolved."""
    frag = url.split("/folder/", 1)[1]
    node_id, key_b64 = frag.split("#", 1)
    key_b64 = key_b64.split("/")[0]
    folder_key = b64d(key_b64)
    res = api({"a": "f", "c": 1, "r": 1, "ca": 1}, node_id)
    nodes = []
    for n in res.get("f", []):
        entry = {"h": n["h"], "p": n.get("p"), "t": n["t"], "s": n.get("s", 0)}
        name = None
        fkey = None
        for cand in node_keys(n, folder_key):
            if n["t"] == 0:                       # file: 32-byte key folds to 16
                k16, iv = unpack_file_key(cand)
                if k16 is None:
                    continue
                at = decrypt_attr(b64d(n.get("a", "")), k16)
                if at:
                    name, fkey, entry["iv"] = at.get("n"), cand, iv
                    break
            else:                                 # folder: key is already 16
                k16 = cand[:16]
                at = decrypt_attr(b64d(n.get("a", "")), k16)
                if at:
                    name, fkey = at.get("n"), cand
                    break
        entry["name"] = name or ("<undecrypted:%s>" % n["h"])
        entry["key"] = fkey
        nodes.append(entry)
    return node_id, folder_key, nodes


def paths_for(nodes):
    """Reconstruct the folder layout so the mirror's tree is preserved."""
    by_h = {n["h"]: n for n in nodes}
    out = {}
    for n in nodes:
        parts, cur, guard = [n["name"]], n.get("p"), 0
        while cur in by_h and guard < 32:
            parts.append(by_h[cur]["name"])
            cur = by_h[cur].get("p")
            guard += 1
        out[n["h"]] = "/".join(reversed(parts[:-1] + [parts[-1]])) if len(parts) > 1 \
            else n["name"]
    return out


def download(node, node_id, dest):
    """Fetch a temporary URL and decrypt the stream with AES-CTR."""
    k16, iv = unpack_file_key(node["key"])
    res = api({"a": "g", "g": 1, "n": node["h"]}, node_id)
    if "g" not in res:
        raise RuntimeError("no download URL for %s" % node["name"])
    ctr = Counter.new(128, initial_value=((iv[0] << 32) + iv[1]) << 64)
    aes = AES.new(k16, AES.MODE_CTR, counter=ctr)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    got = 0
    with requests.get(res["g"], stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(aes.decrypt(chunk))
                    got += len(chunk)
    os.replace(tmp, dest)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    node_id, folder_key, nodes = list_folder(a.url)
    names = paths_for(nodes)
    files = [n for n in nodes if n["t"] == 0 and n["key"]]
    total = sum(n["s"] for n in files)
    print("%d nodes, %d files, %.2f GB" % (len(nodes), len(files), total / 1e9))
    if a.list or not a.out:
        for n in sorted(files, key=lambda x: names[x["h"]]):
            print("  %10.2f MB  %s" % (n["s"] / 1e6, names[n["h"]]))
        undec = [n for n in nodes if n["key"] is None]
        if undec:
            print("  !! %d nodes did not decrypt" % len(undec))
        return

    t0, done = time.time(), 0
    for i, n in enumerate(sorted(files, key=lambda x: names[x["h"]]), 1):
        dest = os.path.join(a.out, names[n["h"]])
        if os.path.exists(dest) and os.path.getsize(dest) == n["s"]:
            print("[skip] %s" % names[n["h"]], flush=True)
            done += n["s"]
            continue
        try:
            got = download(n, node_id, dest)
        except Exception as e:                      # a mirror can drop a stream
            print("[FAIL] %s: %s" % (names[n["h"]], e), flush=True)
            continue
        done += got
        el = time.time() - t0
        print("[ok] %3d/%d  %8.2f MB  %s  (%.0f%%, %.1f min)"
              % (i, len(files), got / 1e6, names[n["h"]],
                 100 * done / max(1, total), el / 60), flush=True)
    print("\ndone: %.2f GB in %.1f min" % (done / 1e9, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
