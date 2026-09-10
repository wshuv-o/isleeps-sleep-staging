"""Fetch canonical BibTeX for every entry in references.bib.

The rule this implements: **a preprint is the last resort.** For each entry we
look for the version of record first -- journal article, then conference
proceedings, then book chapter -- and only fall back to arXiv when no published
form can be found. An entry that already carries a preprint DOI is *not* treated
as resolved; it keeps searching for the published version, which is how
`csbrain2025` was found to have become a NeurIPS paper.

Resolution chain, first hit wins:

1. **CrossRef by DOI** -- if the entry has one, confirm it resolves and that the
   registered title matches. A preprint DOI is recorded but does not stop the search.
2. **CrossRef by title** -- candidates are filtered by title similarity, then
   ranked by publication type (see ``TYPE_RANK``), so a journal article always
   beats a preprint of the same work.
3. **OpenReview** -- for ICLR/NeurIPS papers, which frequently have no DOI at
   all. The venue is taken from the submission invitation, not guessed.
4. **arXiv** -- only if 1-3 all fail.

Every candidate must clear ``--min-sim`` title similarity before it is accepted.
This matters: a bare CrossRef search returns a plausible-looking top hit for
almost any query, and for this bibliography it returned an article about arXiv
moderation policy as the best match for three unrelated papers.

Usage
-----
    python fetch_bib.py                          # report only, writes nothing
    python fetch_bib.py --out references_new.bib # write the fetched entries
    python fetch_bib.py --only labram2024,biot2023
    python fetch_bib.py --mailto you@example.com # CrossRef "polite pool"

Nothing is overwritten in place. Diff the output against references.bib and
merge deliberately -- fetched entries use the publisher's field conventions,
which will not match the hand-written ones exactly.
"""
import argparse
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher

CROSSREF = "https://api.crossref.org/works"
OPENREVIEW = "https://api2.openreview.net/notes/search"
ARXIV = "http://export.arxiv.org/api/query"

# Lower is better. Anything >= PREPRINT_RANK is a fallback, never a first choice.
TYPE_RANK = {
    "journal-article": 0,
    "proceedings-article": 1,
    "book-chapter": 2,
    "book": 3,
    "monograph": 3,
    "report": 4,
    "dissertation": 5,
    "posted-content": 9,
}
PREPRINT_RANK = 9
PREPRINT_DOI = re.compile(r"^10\.(48550|20944|1101|21203|31234)/", re.I)


# --------------------------------------------------------------------- helpers
def norm(s):
    s = re.sub(r"[{}\\]", " ", (s or "").lower())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def similar(a, b):
    """Title agreement, tolerant of the two ways registries mangle a title.

    Registries truncate ("PhysioBank, PhysioToolkit, and PhysioNet" for a title
    that continues "...: Components of a New Research Resource...") and authors
    paraphrase their own titles between preprint and camera-ready. A plain
    sequence ratio scores both as mismatches, so we also measure containment:
    if every word of the shorter title appears in order in the longer one, that
    is a match regardless of the length difference.
    """
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    ratio = SequenceMatcher(None, na, nb).ratio()
    sa, sb = na.split(), nb.split()
    short, long_ = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    i = 0
    for w in long_:
        if i < len(short) and w == short[i]:
            i += 1
    containment = i / len(short) if short else 0.0
    # containment alone is weak evidence on very short titles
    if len(short) < 4:
        containment *= 0.7
    return max(ratio, containment)


def bib_first_surname(fields):
    """First author's family name, or None for institutional/absent authors."""
    raw = fields.get("author") or ""
    first = re.split(r"\s+and\s+", raw)[0].strip()
    first = re.sub(r"[{}\\'\"~^]", "", first).strip()
    if not first or first.lower().startswith(("american ", "the ")):
        return None
    surname = first.split(",")[0].strip() if "," in first else first.split()[-1]
    surname = re.sub(r"[^A-Za-z]", "", surname).lower()
    return surname or None


def author_agrees(fields, cand_surname):
    """False only when both surnames are known and they disagree."""
    ours = bib_first_surname(fields)
    if not ours or not cand_surname:
        return True
    theirs = re.sub(r"[^A-Za-z]", "", cand_surname).lower()
    if not theirs:
        return True
    return ours == theirs or ours in theirs or theirs in ours


def accept(fields, cand_title, cand_surname, min_sim):
    """Title score, gated on the first author. Returns 0.0 when rejected.

    Two papers in this bibliography differ by one word in the title and share a
    topic -- CBraMod and CSBrain both read "A Cross...Brain Foundation Model for
    EEG Decoding" and score 0.76 against each other. Only the author list
    separates them, so a title match alone is never enough.
    """
    sim = similar(fields.get("title", ""), cand_title)
    if sim < min_sim:
        return 0.0
    if not author_agrees(fields, cand_surname):
        # a near-exact title can still win despite an author-list mismatch
        return sim if sim >= 0.95 else 0.0
    return sim


def cr_surname(item):
    a = (item.get("author") or [{}])[0]
    return a.get("family") or a.get("name") or ""


class Http:
    def __init__(self, mailto=None, delay=0.4):
        self.delay = delay
        ua = "MMNet-bib-fetch/1.0 (https://github.com/wshuv-o/isleeps-sleep-staging"
        self.ua = ua + ("; mailto:%s)" % mailto if mailto else ")")

    def get(self, url, accept=None):
        headers = {"User-Agent": self.ua}
        if accept:
            headers["Accept"] = accept
        req = urllib.request.Request(url, headers=headers)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=45) as r:
                    body = r.read().decode("utf-8", "replace")
                time.sleep(self.delay)
                return body
            except urllib.error.HTTPError as e:
                if e.code in (429, 503) and attempt < 2:
                    time.sleep(2 ** attempt * 3)
                    continue
                raise
            except Exception:
                if attempt < 2:
                    time.sleep(2)
                    continue
                raise

    def json(self, url):
        return json.loads(self.get(url))


# ----------------------------------------------------------------- bib parsing
def parse_bib(path):
    txt = io.open(path, encoding="utf-8").read()
    out = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", txt, re.S):
        body = m.group(3)
        fields = {}
        for fm in re.finditer(
                r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*(?=\n\s*\w+\s*=|\s*$)", body, re.S):
            fields[fm.group(1).lower()] = " ".join(fm.group(2).split())
        out.append({"key": m.group(2).strip(), "type": m.group(1).lower(), "fields": fields})
    return out


def rekey(bibtex, key):
    """Replace the fetched entry's key with ours, so \\cite commands keep working."""
    return re.sub(r"^(@\w+\s*\{)[^,]*,", lambda m: m.group(1) + key + ",",
                  bibtex.strip(), count=1)


# -------------------------------------------------------------------- resolvers
def crossref_by_doi(http, fields, doi, min_sim):
    d = http.json("%s/%s" % (CROSSREF, urllib.parse.quote(doi)))["message"]
    ct = (d.get("title") or [""])[0]
    return {"source": "crossref-doi", "doi": d.get("DOI"), "title": ct,
            "type": d.get("type", ""), "venue": (d.get("container-title") or [""])[0],
            "sim": accept(fields, ct, cr_surname(d), min_sim)}


def crossref_by_title(http, entry, min_sim):
    f = entry["fields"]
    venue = f.get("journal") or f.get("booktitle") or ""
    query = urllib.parse.urlencode({
        "query.bibliographic": "%s %s %s" % (f.get("title", ""), venue, f.get("year", "")),
        "rows": 8})
    items = http.json("%s?%s" % (CROSSREF, query))["message"]["items"]
    cands = []
    for b in items:
        ct = (b.get("title") or [""])[0]
        sim = accept(f, ct, cr_surname(b), min_sim)
        if not sim:
            continue
        cands.append({"source": "crossref-search", "doi": b.get("DOI"), "title": ct,
                      "type": b.get("type", ""), "sim": sim,
                      "venue": (b.get("container-title") or [""])[0]})
    # version of record first; among equals, the closest title
    cands.sort(key=lambda c: (TYPE_RANK.get(c["type"], 6), -c["sim"]))
    return cands[0] if cands else None


def openreview(http, entry, min_sim):
    title = entry["fields"].get("title", "")
    q = urllib.parse.urlencode({"term": re.sub(r"[{}]", "", title),
                                "type": "terms", "limit": 5})
    notes = http.json("%s?%s" % (OPENREVIEW, q)).get("notes", [])
    best = None
    for n in notes:
        c = n.get("content", {})

        def val(k):
            v = c.get(k)
            return v.get("value") if isinstance(v, dict) else v

        authors = val("authors") or []
        sim = accept(entry["fields"], val("title"),
                     (authors[0].split()[-1] if authors else ""), min_sim)
        inv = " ".join(n.get("invitations") or [n.get("invitation") or ""])
        # a real venue submission, not a DBLP mirror or a workshop reposting
        if not sim or "/-/Submission" not in inv:
            continue
        if best is None or sim > best["sim"]:
            best = {"source": "openreview", "doi": None, "title": val("title"),
                    "type": "proceedings-article", "sim": sim,
                    "venue": inv.split("/-/")[0].split()[-1],
                    "url": "https://openreview.net/forum?id=%s" % n.get("id"),
                    "bibtex": val("_bibtex")}
    return best


def arxiv(http, entry, min_sim):
    title = re.sub(r"[{}:]", " ", entry["fields"].get("title", ""))
    q = urllib.parse.urlencode({"search_query": 'all:"%s"' % title,
                                "max_results": 5})
    root = ET.fromstring(http.get("%s?%s" % (ARXIV, q)))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for e in root.findall("a:entry", ns):
        ct = (e.findtext("a:title", "", ns) or "").strip()
        au = e.find("a:author/a:name", ns)
        sim = accept(entry["fields"], ct,
                     ((au.text or "").split()[-1] if au is not None and au.text else ""),
                     min_sim)
        if not sim:
            continue
        aid = (e.findtext("a:id", "", ns) or "").rsplit("/", 1)[-1]
        return {"source": "arxiv", "doi": "10.48550/arXiv." + aid.split("v")[0],
                "title": ct, "type": "posted-content", "sim": sim,
                "venue": "arXiv", "url": "https://arxiv.org/abs/" + aid}
    return None


def bibtex_for(http, hit, entry):
    if hit.get("bibtex"):
        return hit["bibtex"]
    if hit.get("doi"):
        try:
            return http.get("%s/%s/transform/application/x-bibtex"
                            % (CROSSREF, urllib.parse.quote(hit["doi"])),
                            accept="application/x-bibtex")
        except Exception:
            pass
        try:
            return http.get("https://doi.org/" + urllib.parse.quote(hit["doi"]),
                            accept="application/x-bibtex")
        except Exception:
            return None
    return None


# ------------------------------------------------------------------------ main
def resolve(http, entry, args):
    """Return (hit, notes). A preprint hit is kept only if nothing better appears."""
    notes, fallback = [], None
    doi = entry["fields"].get("doi", "").strip()
    title = entry["fields"].get("title", "")

    if doi:
        try:
            h = crossref_by_doi(http, entry["fields"], doi, args.min_sim)
            if h["sim"]:
                if PREPRINT_DOI.match(doi) or TYPE_RANK.get(h["type"], 6) >= PREPRINT_RANK:
                    notes.append("existing DOI is a preprint; looking for the published version")
                    fallback = h
                else:
                    return h, notes
            else:
                notes.append("existing DOI resolves but the title differs (%.2f)" % h["sim"])
        except Exception as e:
            notes.append("existing DOI did not resolve: %s" % type(e).__name__)

    for finder, label in ((crossref_by_title, "crossref"), (openreview, "openreview")):
        try:
            h = finder(http, entry, args.min_sim)
        except Exception as e:
            notes.append("%s lookup failed: %s" % (label, type(e).__name__))
            continue
        if not h:
            continue
        if TYPE_RANK.get(h["type"], 6) >= PREPRINT_RANK:
            fallback = fallback or h
            notes.append("%s offered only a preprint; continuing" % label)
            continue
        return h, notes

    try:
        h = arxiv(http, entry, args.min_sim)
    except Exception as e:
        notes.append("arxiv lookup failed: %s" % type(e).__name__)
        h = None
    if h:
        notes.append("no published version found; falling back to arXiv")
        return h, notes
    if fallback:
        notes.append("keeping the preprint record; nothing published found")
    return fallback, notes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bib", default="references.bib")
    p.add_argument("--out", help="write fetched BibTeX here (default: report only)")
    p.add_argument("--report", default="bib_report.md")
    p.add_argument("--only", help="comma-separated cite keys")
    p.add_argument("--min-sim", type=float, default=0.75,
                   help="title-similarity floor for accepting a match (default 0.75)")
    p.add_argument("--mailto", help="contact address for the CrossRef polite pool")
    p.add_argument("--delay", type=float, default=0.4)
    args = p.parse_args(argv)

    entries = parse_bib(args.bib)
    if args.only:
        want = {k.strip() for k in args.only.split(",")}
        entries = [e for e in entries if e["key"] in want]
    http = Http(args.mailto, args.delay)
    print("resolving %d entries (preprints accepted only as a last resort)\n" % len(entries))

    fetched, report = [], []
    counts = {}
    for e in entries:
        key = e["key"]
        hit, notes = resolve(http, e, args)
        if not hit:
            counts["unresolved"] = counts.get("unresolved", 0) + 1
            print("%-26s UNRESOLVED" % key)
            report.append((key, e, None, notes, None))
            continue
        bib = bibtex_for(http, hit, e)
        if bib:
            fetched.append(rekey(bib, key))
        src = hit["source"]
        counts[src] = counts.get(src, 0) + 1
        print("%-26s %-17s %-13s %s" % (key, src, hit["type"][:13],
                                        hit.get("doi") or hit.get("url", "")))
        for n in notes:
            print("%28s- %s" % ("", n))
        report.append((key, e, hit, notes, bib))

    if args.out:
        io.open(args.out, "w", encoding="utf-8").write("\n\n".join(fetched) + "\n")
        print("\nwrote %d entries -> %s" % (len(fetched), args.out))

    with io.open(args.report, "w", encoding="utf-8") as fh:
        fh.write("# Bibliography resolution report\n\n")
        fh.write("Preference order: journal article > proceedings > book chapter > preprint.\n")
        fh.write("Title-similarity floor: %.2f\n\n" % args.min_sim)
        fh.write("| key | source | type | venue | DOI / URL | sim |\n")
        fh.write("|---|---|---|---|---|---|\n")
        for key, e, hit, notes, _ in report:
            if hit:
                fh.write("| `%s` | %s | %s | %s | %s | %.2f |\n" % (
                    key, hit["source"], hit["type"], (hit.get("venue") or "")[:40],
                    hit.get("doi") or hit.get("url", ""), hit["sim"]))
            else:
                fh.write("| `%s` | **unresolved** | | %s | | |\n"
                         % (key, e["fields"].get("journal", "")[:40]))
        flagged = [(k, n) for k, _, _, n, _ in report if n]
        if flagged:
            fh.write("\n## Entries needing a human decision\n\n")
            for key, notes in flagged:
                fh.write("- **`%s`**\n" % key)
                for n in notes:
                    fh.write("  - %s\n" % n)
    print("report -> %s" % args.report)
    print("\nsummary: " + ", ".join("%s=%d" % kv for kv in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
