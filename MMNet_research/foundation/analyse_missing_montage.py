"""The control: does a model that never sees a cardiorespiratory channel also
score lower on the twelve incomplete-montage patients?

If it does, the staging gap belongs to those recordings and not to zero-filling,
because neural-only has no cardiorespiratory input to zero-fill.
"""
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

REPO = r'D:\proc\isleeps-sleep-staging'
F = os.path.join(REPO, 'MMNet_research', 'results', 'revision', 'runs', 'final')
FEAT = os.path.join(REPO, 'data', 'mm_features')

full = json.load(open(os.path.join(F, 'per_subject_seed42.json')))
neur = json.load(open(os.path.join(F, 'neural_only_per_subject.json')))['neural_only|42']['per_subject']

rows = []
for s in full:
    f = os.path.join(FEAT, 'SN%d.npz' % int(s[2:]))
    if not os.path.exists(f) or s not in neur:
        continue
    complete = int(np.load(f)['cvalid'].sum()) == 7
    rows.append((s, complete, full[s]['acc'], neur[s]['acc']))

comp = [r for r in rows if r[1]]
inc = [r for r in rows if not r[1]]
print('patients: %d   complete %d   incomplete %d\n' % (len(rows), len(comp), len(inc)))

print('STAGING ACCURACY BY MONTAGE COMPLETENESS')
print('  %-22s %9s %9s %9s %8s' % ('model', 'complete', 'incomplete', 'gap', 'p'))
print('  ' + '-' * 62)
for idx, name in ((2, 'MM-Net (full)'), (3, 'MM-Net (neural only)')):
    a = np.array([r[idx] for r in comp])
    b = np.array([r[idx] for r in inc])
    p = mannwhitneyu(a, b).pvalue
    print('  %-22s %9.3f %9.3f %+9.3f %8.3f' % (name, a.mean(), b.mean(), b.mean() - a.mean(), p))

print()
print('  Neural-only receives no cardiorespiratory channel at all, so it has')
print('  nothing zero-filled. A gap of the same sign and size in both models')
print('  means the twelve recordings are simply harder.')
print()

# and the direct paired question: on those twelve, is the full model worse than
# neural-only? if zero-filling actively hurt, it would show here.
a = np.array([r[2] for r in inc])
b = np.array([r[3] for r in inc])
print('ON THE TWELVE INCOMPLETE PATIENTS, PAIRED')
print('  full %.4f   neural-only %.4f   diff %+.4f   Wilcoxon p %.3f'
      % (a.mean(), b.mean(), a.mean() - b.mean(), wilcoxon(a, b).pvalue))
a = np.array([r[2] for r in comp])
b = np.array([r[3] for r in comp])
print('ON THE 87 COMPLETE PATIENTS, PAIRED')
print('  full %.4f   neural-only %.4f   diff %+.4f   Wilcoxon p %.3f'
      % (a.mean(), b.mean(), a.mean() - b.mean(), wilcoxon(a, b).pvalue))
