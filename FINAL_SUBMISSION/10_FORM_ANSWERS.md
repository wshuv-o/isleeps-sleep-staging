# Google Form — answers to copy

## Page 3 — Items 2 & 5: The paper and journal reformatting (15 marks)

### Q. Target journal *
Select **Other:** and type:

```
IEEE Journal of Biomedical and Health Informatics (JBHI)
```

### Q. Why does this journal fit your topic and scope? Provide the fund needed for publication. Share the link of the Journal Home page *

```
Target: IEEE Journal of Biomedical and Health Informatics (JBHI).
Home page: https://www.embs.org/jbhi/
IEEE Xplore: https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6221020

JBHI publishes original work "where information and communication technologies
intersect with health, healthcare, life sciences and biomedicine," which is
exactly the register of this paper: a compact multimodal model that reads a full
clinical polysomnogram and returns two clinically actionable outputs at once --
the sleep stage of every 30-second epoch and a per-epoch respiratory-event label
-- validated on a real patient cohort rather than a healthy benchmark set. The
audience is biomedical informatics researchers and clinical engineers working on
physiological time series, who are the people for whom our central finding
matters: deep architectures developed on healthy sleep lose 15-20 accuracy points
on subacute ischemic stroke patients, while a physiologically grounded feature
model does not. That is a deployment constraint for anyone applying sleep AI to a
clinical population, and JBHI is where such a result is read and acted on. Our
clinical-facing analyses -- association with scored AHI (rho = 0.315, p = 0.0017,
n = 96), staging stratified by AASM severity band, and per-event-type detection
(hypopnea 0.692, obstructive 0.763, central 0.840) -- are the kind JBHI reviewers
expect and that a general machine-learning venue would treat as out of scope.
Sleep staging and sleep-disordered-breathing detection both have established
precedent in the journal.

Funding needed for publication: JBHI is a hybrid journal. Under the standard
subscription (non-open-access) route there is no mandatory article processing
charge, so the paper can be published at no cost to the authors. If open access
is elected, IEEE's hybrid open-access charge applies -- approximately USD 2,195
-- and overlength page charges may apply beyond the standard page allowance. We
would submit under the no-fee subscription route, so no publication fund is
required; open access would be elected only if institutional funding were
available.
```

> **Verify before submitting:** the exact APC is not published on the JBHI home
> page. Check the current figure under IEEE's author guidelines and adjust the
> number if it has changed.

### Q. Confirm the conversion covered *
Tick **all seven**:

- [x] Class file / template
- [x] Section structure
- [x] Reference style
- [x] Figure and table placement
- [x] Abstract style
- [x] Author block
- [x] Required declarations

**How each is satisfied** (keep this to hand in case you are asked):

| Item | What was done |
|---|---|
| Class file / template | `\documentclass[journal]{IEEEtran}` — the class JBHI requires for submission. Confirmed against the target journal's author requirements. |
| Section structure | Restructured to the required order and **new sections added**: III Critical Gaps and Limitations, VII-A Limitations, VII-B Future Work. Results expanded from 2 subsections to 10. |
| Reference style | IEEE numeric style throughout; bibliography rebuilt and expanded from 11 to 30 entries, 27 of them 2023 or later. |
| Figure and table placement | All floats moved to two-column `table*`/`figure*` placement with captions below figures and above tables, per IEEE style. Tables grew from 2 to 8. |
| Abstract style | Rewritten as a single unstructured paragraph with the headline metrics stated, per IEEE format, and an IEEE keyword block added. |
| Author block | IEEE `\author` block with all four authors, institutional affiliation, per-author student emails, and a designated corresponding author. |
| Required declarations | Added data availability, code availability, ethics (NIMHANS Institutional Ethics Committee), and funding statements. |

### Q. Upload — Paper PDF, FRESH copy *
```
02c_Paper_FRESH_final.pdf        (11 pages, 8.8 MB)
```

### Q. Upload — Same paper with all changes highlighted *
```
02b_Paper_HIGHLIGHTED_changes.pdf   (12 pages, 8.8 MB)
```

### Q. Upload — the old copy
```
02a_Paper_OLD_version.pdf        (5 pages, 8.1 MB)
```

All three are under the 10 MB per-file limit.

---

## Note on the highlighted copy

It was produced with `latexdiff` against the first committed version of the
manuscript, recovered from the repository's git history (commit `e9fbb5b`).
**159 additions and 106 deletions** are marked in the body text. Table and figure
internals are not individually marked, because the markup is illegal inside
`booktabs` rules and prevents the document compiling; the tables are almost
entirely new in any case, and every prose change is highlighted.

The change-tracking process is reproducible by anyone on the team:

```bash
git show e9fbb5b:paper/multimodal.tex > OLD_v1.tex
latexdiff --config="PICTUREENV=picture|tabular|tabularx|array|align|equation" \
          --append-safecmd="toprule,midrule,bottomrule,multirow,multicolumn" \
          -t CFONT OLD_v1.tex multimodal.tex > multimodal_diff.tex
pdflatex multimodal_diff.tex
```
