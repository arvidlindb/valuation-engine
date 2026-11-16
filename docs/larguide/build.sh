#!/bin/sh
# Bygger docs/larguide.pdf från LaTeX-källorna i den här katalogen.
# Kräver tectonic (https://tectonic-typesetting.github.io), som laddar ned de paket som behövs.
# Användning: docs/larguide/build.sh
set -eu
cd "$(dirname "$0")"
TECTONIC="${TECTONIC:-tectonic}"
mkdir -p build
"$TECTONIC" --keep-logs --outdir build larguide.tex
cp build/larguide.pdf ../larguide.pdf
echo "Skrev $(cd .. && pwd)/larguide.pdf"
