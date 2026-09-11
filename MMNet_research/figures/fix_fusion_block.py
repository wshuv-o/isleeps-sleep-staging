"""Rebuild Figure 2's fusion block to match the model that produced the numbers.

The block drew a four-head cross-attention transformer: tokenize, softmax(QK'/sqrt d)V,
concat.Wo, residual, LayerNorm, Feed-Forward. That is MMFeatureNet's fusion="cross"
branch (mmnet_core.py:101 CrossFusion). The published model calls
run_10fold(fusion="concat") -> mmnet_core.py:125:

    self.fuse = nn.Sequential(nn.Linear(d + d_card, d), nn.GELU(), nn.Dropout(drop))
    fz = self.fuse(torch.cat([e, c], -1))                      # 128 + 64 -> 128

Verified by reconstructing the exact final model: 2,764,774 trainable parameters
(matching the paper), 0 MultiheadAttention modules, fuse = Linear(192->128).

Group 98 holds the block contents; container 13 and title 135 frame it. Both keep
their geometry so the arrows in and out are untouched; only the contents change.
Cell styles are copied verbatim from the EEG encoder row (group 27) so the new
blocks use the figure's existing visual language and legend colours.
"""
import io
import os
import re
import xml.etree.ElementTree as ET

SP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SP, 'fig_arch.drawio')
DST = os.path.join(SP, 'fig_architecture_fixed.drawio')

raw = io.open(SRC, encoding='utf-8').read()

# The XML came out of a PDF string, where ( ) and \ are backslash-escaped.
before = (raw.count('\\('), raw.count('\\)'), raw.count('\\\\'))
raw = re.sub(r'\\([()\\])', r'\1', raw)
print('unescaped PDF string artifacts  \\( \\) \\\\ :', before)

root = ET.fromstring(raw)
model = root.find('.//mxGraphModel')
croot = model.find('root')
cells = croot.findall('mxCell')
byid = {c.get('id'): c for c in cells}

# --- 1. remove the attention machinery -----------------------------------
doomed = [c for c in cells if c.get('parent') == '98']
print('removing %d cells from group 98' % len(doomed))
for c in doomed:
    croot.remove(c)

# --- 2. styles lifted verbatim from the EEG encoder row ------------------
CUBE = ('shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;'
        'darkOpacity=0.06;darkOpacity2=0.12;fillColor=%s;strokeColor=#2B2F33;size=9;')
TEXT = 'text;html=1;align=center;verticalAlign=middle;fontSize=8;'
EDGE = ('edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;'
        'strokeColor=#5A6570;strokeWidth=1.0;exitX=1;exitY=0.5;exitDx=0;exitDy=0;'
        'entryX=0;entryY=0.5;entryDx=0;entryDy=0;')
GREY, GREEN, ORANGE = '#C8CCD2', '#3FAE5A', '#F0A03C'   # input / linear / output

# group 98 is 286 x 177; three blocks centred in it
BLOCKS = [
    # id,   x,    w,    fill,   dim,     caption
    ('200', 44.0, 16.0, GREY,   '192', 'concat [e ; c]'),
    ('203', 128.0, 16.0, GREEN, '128', 'Linear W<sub>f</sub>'),
    ('206', 212.0, 16.0, ORANGE, '128', 'GELU + Dropout'),
]
CUBE_Y, CUBE_H = 50.0, 78.0
LBL_W = 56.0


def cell(cid, value, style, x, y, w, h, parent='98', vertex='1'):
    e = ET.SubElement(croot, 'mxCell')
    e.set('id', cid)
    if value is not None:
        e.set('value', value)
    e.set('style', style)
    e.set('parent', parent)
    e.set('vertex', vertex)
    g = ET.SubElement(e, 'mxGeometry')
    for k, v in (('x', x), ('y', y), ('width', w), ('height', h)):
        g.set(k, str(v))
    g.set('as', 'geometry')
    return e


for cid, x, w, fill, dim, cap in BLOCKS:
    cx = x + w / 2.0
    cell(cid, None, CUBE % fill, x, CUBE_Y, w, CUBE_H)
    cell(str(int(cid) + 1), '<b>%s</b>' % dim, TEXT, cx - LBL_W / 2, 30.0, LBL_W, 14.0)
    cell(str(int(cid) + 2), cap, TEXT, cx - LBL_W / 2, 133.0, LBL_W, 24.0)

for n, (a, b) in enumerate((('200', '203'), ('203', '206'))):
    e = ET.SubElement(croot, 'mxCell')
    e.set('id', '21%d' % n)
    e.set('style', EDGE)
    e.set('parent', '98')
    e.set('edge', '1')
    e.set('source', a)
    e.set('target', b)
    g = ET.SubElement(e, 'mxGeometry')
    g.set('relative', '1')
    g.set('as', 'geometry')

# --- 3. retitle: the block is still cross-modal, but it is not attention --
t = byid['135']
old = t.get('value')
t.set('value', re.sub(r'Cross-modal fusion', 'Multimodal fusion', old))
print('title: %r -> %r' % (re.sub(r'<[^>]+>', '', old).strip(),
                           re.sub(r'<[^>]+>', '', t.get('value')).strip()))

io.open(DST, 'w', encoding='utf-8', newline='\n').write(
    ET.tostring(root, encoding='unicode'))
print('wrote', DST)
