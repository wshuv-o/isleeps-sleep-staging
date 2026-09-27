"""Items 7i, 7j: parameter counts of the variants the paper names (CPU, no data load).

The model classes are taken verbatim from model/mmnet_core.py (class definitions
only, so the module's data loading does not run) and foundation/cardio_cnn.py.
The neural-only row of Table 4 is produced by run_neural_only_pcf.py as the full
model with card_drop=("all",): the cardiorespiratory input is zeroed and every
module is still present, so its count equals the full model's.

  python param_counts.py
"""
import importlib.util
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import _common as C  # noqa: E402

src = open(os.path.join(C.MM, "model", "mmnet_core.py"), encoding="utf8").read()
ns = {"nn": nn, "torch": torch}
exec(src[src.index("class FeatMLP"):src.index('print("parameters (concat)')], ns)
spec = importlib.util.spec_from_file_location("cc", os.path.join(C.MM, "foundation", "cardio_cnn.py"))
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
M, CNN = ns["MMFeatureNet"], cc.CardioCNN


def P(m):
    return int(sum(p.numel() for p in m.parameters()))


def build(fusion="concat", cnn=True):
    m = M(n_eeg=388, n_card=14, hidden=256, fusion=fusion)
    if cnn:
        m.card_enc = CNN(d=64)
    return m


full = build()
res = {
    "full_model (published, 388-d neural + CardioCNN)": P(full),
    "cardio_cnn": P(CNN(d=64)),
    "neural_only as run (full model, cardio input zeroed)": P(full),
    "healthy Sleep-EDF row as run (same network, cardio zeroed)": P(full),
    "hypothetical: fusion over neural only, CNN removed": P((lambda m: (setattr(m, "card_enc", nn.Identity()), m)[1])(build("none"))),
    "hypothetical: concat kept, CNN removed": P((lambda m: (setattr(m, "card_enc", nn.Identity()), m)[1])(build())),
    "earlier feature-only model (14-d cardio MLP, same widths)": P(build(cnn=False)),
}
C.save("param_counts.json", res)
for k, v in res.items():
    print("%-62s %s" % (k, format(v, ",")))
