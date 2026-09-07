"""Hyperparameter and architecture sweep -- the search this project never ran.

Two facts motivate this:

  * 13_architecture_ablation found that DELETING the BiLSTM raises staging from
    0.7238 to 0.7386 across three seeds (+0.0148, about one fold SD). A published
    architecture that improves when a component is removed has not been optimised.
  * Nothing in this repository tunes learning rate, dropout, width, depth or
    weight decay. The published values appear to be a first guess.

The learning curve in the paper shows that more PATIENTS do not improve staging.
That is a statement about data volume, not about modelling, and it has been
repeatedly read as though it were the latter. This module exists to test the
modelling half properly.

train_fold hardcodes lr and weight decay and builds MMFeatureNet without passing
architecture arguments, so both are patched by recompiling the function's own
source with the literals substituted -- the same technique arms.py uses, with the
same asserted substitution counts so a future edit to mmnet_core fails loudly
rather than silently sweeping nothing.
"""
import inspect
import re
import textwrap

import arms

# literal -> how many times it must appear in train_fold's source
_TRAIN_LITERALS = {"lr": (r"lr=1e-3", 1), "wd": (r"weight_decay=1e-4", 1)}


def _patched_train_fold(C, lr, wd):
    src = textwrap.dedent(inspect.getsource(C.train_fold))
    for name, (pat, expected) in _TRAIN_LITERALS.items():
        repl = {"lr": "lr=%g" % lr, "wd": "weight_decay=%g" % wd}[name]
        src, n = re.subn(pat, repl, src)
        if n != expected:
            raise RuntimeError("train_fold: replaced %d of %r, expected %d -- "
                               "mmnet_core changed, re-check the sweep patch"
                               % (n, pat, expected))
    ns = dict(C.__dict__)
    exec(compile(src, "<train_fold:lr=%g,wd=%g>" % (lr, wd), "exec"), ns)
    return ns["train_fold"]


class config:
    """Run mmnet_core with one arm AND one hyperparameter configuration.

        with config(C, "pretrained", hidden=256, drop=0.15, lr=3e-4):
            r = C.run_10fold(fusion="concat", temporal="none", seed=42)

    Restores everything on exit. `temporal` is passed to run_10fold directly
    because mmnet_core already threads it through.
    """

    def __init__(self, C, variant, d=128, hidden=128, layers=2, drop=0.3,
                 lr=1e-3, wd=1e-4):
        self.C, self.variant = C, variant
        self.hp = dict(d=d, hidden=hidden, layers=layers, drop=drop, lr=lr, wd=wd)

    def __enter__(self):
        C = self.C
        self._arm = arms.arm(C, self.variant)
        self._arm.__enter__()
        self.dim = self._arm.dim
        self._net, self._train = C.MMFeatureNet, C.train_fold
        net, hp, dim = self._net, self.hp, self.dim

        def make(**kw):
            kw.setdefault("d", hp["d"])
            kw.setdefault("hidden", hp["hidden"])
            kw.setdefault("layers", hp["layers"])
            kw.setdefault("drop", hp["drop"])
            # arms.arm already bakes n_eeg in for embedding arms
            return net(**kw)

        C.MMFeatureNet = make
        C.train_fold = _patched_train_fold(C, hp["lr"], hp["wd"])
        return self

    def __exit__(self, *exc):
        self.C.MMFeatureNet, self.C.train_fold = self._net, self._train
        self._arm.__exit__(*exc)
        return False

    def label(self):
        return "%s|t=%s|d%d|h%d|l%d|dr%.2f|lr%g|wd%g" % (
            self.variant, "?", self.hp["d"], self.hp["hidden"], self.hp["layers"],
            self.hp["drop"], self.hp["lr"], self.hp["wd"])
