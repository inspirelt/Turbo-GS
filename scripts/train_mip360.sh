#!/bin/bash
# Train the nine Mip-NeRF 360 scenes in turn at -r 1 -i images --eval; each goes to <output_root>/<scene>.
# usage: bash scripts/train_mip360.sh <data_root> <output_root> [--preset 10k|15k|25k]
set -eu
usage(){ echo "usage: bash scripts/train_mip360.sh <data_root> <output_root> [--preset 10k|15k|25k]" >&2; exit 2; }
[ $# -eq 2 ] || [ $# -eq 4 ] || usage
DATA=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
PRESET=()
if [ $# -eq 2 ]; then [ "$1" = --preset ] || usage; PRESET=(--preset "$2"); fi
cd "$(dirname "$0")/.."
for scene in bicycle bonsai counter flowers garden kitchen room stump treehill; do
    echo "=== $scene $(date)"
    python train.py -s "$DATA/$scene" -m "$OUT/$scene" --eval -i images -r 1 "${PRESET[@]}"
done
echo "=== all nine scenes done $(date)"
