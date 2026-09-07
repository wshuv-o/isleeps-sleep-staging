#!/usr/bin/env bash
# CBraMod is third-party (Apache-2.0, wjq-learning/CBraMod) and is NOT vendored here.
# Fetch the model code and released weights into this directory before use.
set -eu
cd "$(dirname "$0")"
mkdir -p models && : > models/__init__.py
for f in cbramod.py criss_cross_transformer.py; do
  curl -sSLf "https://raw.githubusercontent.com/wjq-learning/CBraMod/main/models/$f" -o "models/$f"
done
curl -sSLf "https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth" -o cbramod.pth
echo "fetched: models/cbramod.py, models/criss_cross_transformer.py, cbramod.pth ($(du -h cbramod.pth | cut -f1))"
