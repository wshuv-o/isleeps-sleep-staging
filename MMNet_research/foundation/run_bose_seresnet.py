"""Bose et al.'s SE-ResNet-18 + BiLSTM, from their released code, on our folds.

The referee asks for the iSLEEPS model of Bose et al. (ISBI 2026, arXiv:2603.23582),
reported at 74.7 / 67.7 / 0.64, to be "re-run on your folds". We do not quote that
figure, because the released code does not produce a number comparable with ours:

  github.com/HimalayanSaswataBose/iSLEEPS_GeneralisationGapAndExplainability
  (commit f9eb0cd, model_train_gradcam.ipynb)

  * the checkpoint of each fold is chosen on validation macro-F1, and the metrics
    the notebook logs are that same validation fold's -- the epoch is selected on
    the data it is scored on;
  * the held-out 20% test set is then used to pick the best of the ten fold
    models (`if f1 > best_f1`), so the test set also does selection;
  * windows are cut at stride 4 and loaders use drop_last=True, so roughly a
    quarter of the epochs are ever scored, not every epoch of every night;
  * all 100 recordings are used, including SN28, a byte-identical copy of SN15,
    which can land on both sides of a split.

What is taken verbatim from the notebook: the SE-ResNet-18 encoder (BasicBlock with
kernel-7 convolutions, SE reduction 16, block dropout 0.2, layers [2,2,2,2], stem
kernel 15 stride 2, zero-initialised residual BN), the 512-d embedding, the six-layer
bidirectional LSTM with hidden size 100 and dropout 0.1 (the paper says three; the
code builds six, `NUM_LAYERS = 6`), the read-out at window index ceil(9/2) = 5, the
2000-200-5 head with no activations between layers, log-softmax + NLL loss with no
class weighting, Adam at 1e-3, batch 32, 10 epochs, 9-epoch windows at stride 4 for
training, single-channel C4:M1.

What changes, all toward the protocol every other row in the table uses:

  * our ten patient-independent folds (mmnet_core.FOLDS, N = 99, SN28 excluded),
    the checkpoint chosen on a validation split of training patients only;
  * every epoch of every test night is scored (one window per epoch, edges padded
    by repeating the first/last epoch);
  * the training loader is shuffled -- the notebook's is not, which trains on
    windows in recording order; shuffling can only help their model;
  * per-recording z-scoring of the EEG, since their preprocessing is not released.

Results are reported both raw (as the authors decode) and with the HMM Viterbi
decode the other rows use.

  KMP_DUPLICATE_LIB_OK=TRUE python run_bose_seresnet.py [--seeds 42] [--folds 10]
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import mmnet_core as C          # noqa: E402
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score  # noqa: E402

P7 = os.path.join(REPO, "data", "processed7")
OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final",
                   "bose_seresnet.json")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# verbatim from the notebook
WINDOW, STRIDE, BATCH, EPOCHS, LR = 9, 4, 32, 10, 1e-3
EMB, HIDDEN, LSTM_LAYERS, DROPOUT = 512, 100, 6, 0.1
MID = math.ceil(WINDOW / 2)     # = 5: the notebook reads out index 5, not the centre 4
EEG_CH = 0                      # C4:M1

# The paper's folds are mmnet_core.make_folds over the 99 subjects (processed7 minus
# the SN28 duplicate). mmnet_core.FOLDS is built from data/mm_features instead, which
# on this machine is missing SN2, SN13 and SN17 -- it would give 96-subject folds
# with a different assignment -- so the folds are rebuilt from processed7 here.
SUBJECTS = sorted(int(os.path.basename(p)[2:-4]) for p in glob.glob(os.path.join(P7, "SN*.npz"))
                  if int(os.path.basename(p)[2:-4]) not in C.DUP)
FOLDS = C.make_folds(SUBJECTS)


# ------------------------------------------------------------- their model
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(channel // reduction, channel, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        b, c, _ = x.size()
        return x * self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1)


def conv7(i, o, stride=1):
    return nn.Conv1d(i, o, kernel_size=7, stride=stride, padding=3, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1, self.bn1 = conv7(inplanes, planes, stride), nn.BatchNorm1d(planes)
        self.conv2, self.bn2 = conv7(planes, planes), nn.BatchNorm1d(planes)
        self.relu, self.relu2 = nn.ReLU(inplace=True), nn.ReLU(inplace=True)
        self.se, self.downsample, self.dropout = SELayer(planes), downsample, nn.Dropout(.2)

    def forward(self, x):
        out = self.dropout(self.relu(self.bn1(self.conv1(x))))
        out = self.se(self.bn2(self.conv2(out)))
        identity = self.downsample(x) if self.downsample is not None else x
        return self.relu2(out + identity)


class ResNet(nn.Module):
    def __init__(self, layers=(2, 2, 2, 2), in_channel=1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(in_channel, 64, 15, stride=2, padding=7, bias=False)
        self.bn1, self.relu = nn.BatchNorm1d(64), nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, planes, blocks, stride=1):
        down = None
        if stride != 1 or self.inplanes != planes:
            down = nn.Sequential(nn.Conv1d(self.inplanes, planes, 1, stride, bias=False),
                                 nn.BatchNorm1d(planes))
        layers = [BasicBlock(self.inplanes, planes, stride, down)]
        self.inplanes = planes
        layers += [BasicBlock(planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.avgpool(x).flatten(1)


class SSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = ResNet()
        self.lstm = nn.LSTM(EMB, HIDDEN, LSTM_LAYERS, dropout=DROPOUT, batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc1, self.fc2 = nn.Linear(2 * HIDDEN, 2000), nn.Linear(2000, 200)
        self.fc3 = nn.Linear(200, 5)

    def embed(self, x):                       # [N, 3000] -> [N, 512]
        return self.cnn(x.unsqueeze(1))

    def head(self, e):                        # [B, WINDOW, 512] -> log-probs [B, 5]
        h, _ = self.lstm(e)                   # zero initial state, as in the notebook
        out = self.dropout(h[:, MID])
        return torch.log_softmax(self.fc3(self.fc2(self.fc1(out))), -1)

    def forward(self, x):                     # [B, WINDOW, 3000]
        b, t, n = x.shape
        return self.head(self.embed(x.reshape(b * t, n)).reshape(b, t, -1))


# ------------------------------------------------------------------- data
def load():
    data = {}
    for f in sorted(glob.glob(os.path.join(P7, "SN*.npz")),
                    key=lambda p: int(os.path.basename(p)[2:-4])):
        sid = int(os.path.basename(f)[2:-4])
        if sid not in SUBJECTS:
            continue
        d = np.load(f, allow_pickle=True)
        x = np.asarray(d["x"][:, EEG_CH], dtype=np.float32)
        x = np.nan_to_num((x - x.mean()) / (x.std() + 1e-6))
        data[sid] = (x, d["y"].astype(np.int64))
    return data


def eval_index(n):
    """One window per epoch t with t at position MID, edges padded -> [n, WINDOW]."""
    return np.clip(np.arange(n)[:, None] + np.arange(WINDOW)[None, :] - MID, 0, n - 1)


def run_fold(data, tr, va, te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    # training windows exactly as the notebook cuts them: stride 4, label at index 5
    wins = np.array([(si, i * STRIDE) for si, s in enumerate(tr)
                     for i in range((len(data[s][1]) - WINDOW) // STRIDE + 1)], dtype=np.int64)
    xs = [torch.from_numpy(data[s][0]) for s in tr]
    ys = [data[s][1] for s in tr]

    model = SSModel().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.NLLLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")

    @torch.no_grad()
    def probs(subs):
        """Eval: the encoder is per-epoch and BN is frozen, so embed each epoch once
        and assemble the windows from embeddings -- identical to windowing raw input."""
        model.eval(); out = {}
        for s in subs:
            x = torch.from_numpy(data[s][0])
            with torch.amp.autocast("cuda", enabled=DEV == "cuda"):
                e = torch.cat([model.embed(x[i:i + 512].to(DEV)).float()
                               for i in range(0, len(x), 512)])
            wi = torch.from_numpy(eval_index(len(x))).to(DEV)
            p = [model.head(e[wi[i:i + 1024]]).exp().cpu().numpy()
                 for i in range(0, len(wi), 1024)]
            out[s] = np.concatenate(p)
        return out

    best, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(len(wins))
        for i in range(0, len(perm) - BATCH + 1, BATCH):          # drop_last, as theirs
            sel = wins[perm[i:i + BATCH]]
            xb = torch.stack([xs[si][st:st + WINDOW] for si, st in sel]).to(DEV)
            yb = torch.tensor([ys[si][st + MID] for si, st in sel], device=DEV)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=DEV == "cuda"):
                out = model(xb)
            loss = crit(out.float(), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        pv = probs(va)
        yv = np.concatenate([data[s][1] for s in va])
        vf1 = f1_score(yv, np.concatenate([pv[s] for s in va]).argmax(1), average="macro")
        print("    epoch %2d  val mF1 %.4f" % (ep + 1, vf1), flush=True)
        if vf1 > best:                        # their ModelCheckpoint(monitor="vF1")
            best = vf1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    Am = np.ones((C.NC, C.NC)); pi = np.ones(C.NC)
    for s in tr:
        y = data[s][1]; pi[y[0]] += 1
        for a, b in zip(y[:-1], y[1:]):
            Am[a, b] += 1
    A_log = np.log(Am / Am.sum(1, keepdims=True)); pi_log = np.log(pi / pi.sum())
    pt = probs(te)
    yt = np.concatenate([data[s][1] for s in te])
    raw = np.concatenate([pt[s].argmax(1) for s in te])
    hmm = np.concatenate([C.hmm(A_log, pi_log, np.log(pt[s] + C.EPS)) for s in te])
    return yt, raw, hmm


def metrics(yt, yp):
    return dict(acc=float(accuracy_score(yt, yp)),
                mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                kappa=float(cohen_kappa_score(yt, yp)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    data = load()
    n_par = sum(p.numel() for p in SSModel().parameters())
    print("Bose et al. SE-ResNet-18 + 6-layer BiLSTM | %d subjects | %d folds x %d seeds"
          " | %.2f M params | %s" % (len(data), a.folds, len(a.seeds), n_par / 1e6, DEV),
          flush=True)

    per_seed, t0 = {}, time.time()
    for seed in a.seeds:
        rows = []
        for fi, (tr_all, te) in enumerate(FOLDS[:a.folds]):
            rng = np.random.RandomState(100 + fi)
            tr_all = [s for s in tr_all if s in data]
            rng.shuffle(tr_all)
            nv = max(8, len(tr_all) // 9)
            va, tr = tr_all[:nv], tr_all[nv:]
            te = [s for s in te if s in data]
            yt, raw, hmm = run_fold(data, tr, va, te, seed)
            r = dict(fold=fi, n=int(len(yt)), raw=metrics(yt, raw), hmm=metrics(yt, hmm))
            rows.append(r)
            print("  seed %-3d fold %d/%d  raw acc %.4f mF1 %.4f k %.4f | hmm acc %.4f "
                  "mF1 %.4f k %.4f  [%.1f min]"
                  % (seed, fi + 1, a.folds, r["raw"]["acc"], r["raw"]["mf1"],
                     r["raw"]["kappa"], r["hmm"]["acc"], r["hmm"]["mf1"], r["hmm"]["kappa"],
                     (time.time() - t0) / 60), flush=True)
            per_seed["seed|%d" % seed] = rows
            json.dump(dict(partial=True, per_seed=per_seed), open(a.out, "w"), indent=1)

    summary = {}
    for dec in ("raw", "hmm"):
        summary[dec] = {}
        for m in ("acc", "mf1", "kappa"):
            v = np.array([r[dec][m] for rows in per_seed.values() for r in rows])
            summary[dec][m] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), n=int(v.size))
    json.dump(dict(
        model="SE-ResNet-18 + BiLSTM (Bose et al., ISBI 2026), architecture from the "
              "authors' released notebook",
        source="github.com/HimalayanSaswataBose/iSLEEPS_GeneralisationGapAndExplainability "
               "@ f9eb0cd, model_train_gradcam.ipynb",
        reported_by_authors=dict(acc=0.747, mf1=0.677, kappa=0.64),
        protocol="10-fold patient-independent, mmnet_core.make_folds over the 99 subjects, N=%d, seeds %s, "
                 "checkpoint on a validation split of training patients, every test "
                 "epoch scored" % (len(data), a.seeds),
        input="C4:M1, 9-epoch window, read-out at index %d" % MID,
        params_millions=round(n_par / 1e6, 3), per_seed=per_seed, summary=summary,
        minutes=(time.time() - t0) / 60), open(a.out, "w", encoding="utf-8"), indent=1)
    for dec in ("raw", "hmm"):
        s = summary[dec]
        print("SUMMARY %-3s acc %.4f +- %.4f | mF1 %.4f | kappa %.4f"
              % (dec, s["acc"]["mean"], s["acc"]["sd"], s["mf1"]["mean"], s["kappa"]["mean"]))
    print("authors report 0.747 / 0.677 / 0.64 on their own protocol\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
