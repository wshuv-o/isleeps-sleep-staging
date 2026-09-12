# Publishing to the public MM-Net repository

`github.com/wshuv-o/MM-Net` is **public**. It is the repository cited in the paper's
Data and Code Availability section, so reviewers, editors and readers will open it.
This working repository is where the mess lives; that one is where the work is shown.

Anything pushed there is permanent in a way that anything here is not. A commit
message can be amended only until someone has pulled it, and a public repository is
indexed, mirrored and cloned without warning.

---

## Before every push to MM-Net

**Read the commit message as a stranger would.** It appears under the author's name
on a page attached to a peer-reviewed paper. It should read like something a
researcher wrote about their own work: what changed, and why it matters. Not a
progress note, not a conversation, not a to-do list.

**No AI attribution anywhere.** No `Co-Authored-By` trailer naming an assistant, no
mention of Claude, Anthropic, Copilot or any other tool in a commit message, a
description, a code comment or the README. Authorship of this work belongs to the
named authors. Check with:

```bash
git log --format=%B | grep -i 'co-authored-by\|claude\|anthropic\|copilot'
```

That should print nothing. It is already clean; keep it that way.

**No working notes.** Phrases like "verified before committing", "this was wrong
until now", "caught this by luck", or an account of what went wrong while making the
change belong in this repository, not in that one. State what the commit does.

**Nothing about the review.** Reviewer comments, the responses to them, the number
of revision rounds, and internal assessments of the manuscript are not public
material while the paper is under review.

**The manuscript itself is not published there.** The PDF and the LaTeX source stay
here. MM-Net carries the code, results and figures behind the paper, not the paper.

---

## Checklist

```bash
# from the MM-Net working copy
git log --format=%B | grep -i 'co-authored-by\|claude\|anthropic'   # expect nothing
git ls-files | grep -i 'multimodal_access\.\(pdf\|tex\)'            # expect nothing
git log -1 --format=%B                                             # read it as a stranger
git status --porcelain                                             # expect nothing unexpected
git push
```

If a bad message does reach the remote, amend and `git push --force-with-lease`
immediately. That rewrites history safely while nobody else has pulled, and stops
being an option once someone has.

---

## The two repositories

| | |
|---|---|
| `isleeps-sleep-staging` | this one. Working history, manuscript, reviews, submission packages, experiments that went nowhere. |
| `MM-Net` | public. Code, results and figures for what the paper reports, and nothing else. |
