#!/bin/bash
# Apply patches/musl to an ExecuTorch checkout and to the PyTorch headers its build includes.
#
#   tools/apply-musl-patches.sh <executorch-src> [<torch-include-dir>]
#
# <executorch-src>     ExecuTorch v1.5.1 checkout (patches/musl/executorch, against 3b60683).
# <torch-include-dir>  the include/ directory of the torch package the ExecuTorch build finds
#                      (<site-packages>/torch/include of the torch wheel, or a PyTorch source tree laid
#                      out the same way); patches/musl/pytorch, against PyTorch cf30153 (v2.13.0).
#                      Omitted: the PyTorch patches are skipped.
# Idempotent: a patch already applied (forward) is left alone; one that applies neither way fails.
set -euo pipefail
P=$(cd "$(dirname "$0")/../patches/musl" && pwd)
ET=$1
TI=${2:-}

# apply_git <checkout> <patch>
apply_git() {
  if git -C "$1" apply --check -R "$2" 2>/dev/null; then
    echo "already applied: $(basename "$2")"
  elif git -C "$1" apply --check "$2"; then
    git -C "$1" apply "$2"
    echo "applied: $(basename "$2")"
  else
    echo "does not apply: $2 in $1" >&2
    exit 1
  fi
}
# apply_patch <dir> <strip> <patch>: plain patch(1), since the torch package is not a git checkout.
apply_patch() {
  if patch -d "$1" -p"$2" -R -f -s --dry-run < "$3" > /dev/null 2>&1; then
    echo "already applied: $(basename "$3")"
  elif patch -d "$1" -p"$2" -N -f -s --dry-run < "$3"; then
    patch -d "$1" -p"$2" -N -f -s --no-backup-if-mismatch < "$3"
    echo "applied: $(basename "$3")"
  else
    echo "does not apply: $3 in $1" >&2
    exit 1
  fi
}

for f in "$P"/executorch/*.patch; do apply_git "$ET" "$f"; done
if [ -n "$TI" ]; then
  # The wheel installs torch/headeronly/... at include/torch/headeronly/... (strip 1) and
  # aten/src/ATen/... at include/ATen/... (strip 3).
  for f in "$P"/pytorch/*.patch; do
    if grep -q '^+++ b/aten/src/' "$f"; then apply_patch "$TI" 3 "$f"; else apply_patch "$TI" 1 "$f"; fi
  done
fi
