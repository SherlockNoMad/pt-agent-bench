#!/usr/bin/env bash
# Build the (gitignored) workspace: a pytorch clone + a conda env with all build deps.
# Workers clone this env and rsync-copy this tree per task. See docs/ for the full rationale.
#
#   bash setup_workspace.sh
#
# Override location with PTAB_WORKSPACE (must match config.py); defaults to ./workspace.
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
WS="${PTAB_WORKSPACE:-$REPO/workspace}"
SRC="$WS/src"; ENV="$WS/conda-env"
mkdir -p "$WS"

echo "== [1/3] clone pytorch/pytorch (blobless) -> $SRC =="
if [ ! -d "$SRC/.git" ]; then
  git clone --filter=blob:none https://github.com/pytorch/pytorch.git "$SRC"
fi
git -C "$SRC" submodule update --init --recursive

echo "== [2/3] create conda env (python 3.11) + build deps -> $ENV =="
if [ ! -d "$ENV" ]; then
  conda create -y -p "$ENV" python=3.11
fi
# NOTE: pytest MUST be 7.4.x — pytest 8 breaks pytorch's test/conftest.py (removed `path` hook).
"$ENV/bin/pip" install -q cmake==3.31.6 ninja pyyaml typing_extensions numpy setuptools wheel \
    requests expecttest hypothesis "pytest==7.4.4"
"$ENV/bin/pip" install -q -r "$SRC/requirements.txt" || true

echo "== [3/3] first CPU-only build (BUILD_TEST=1) =="
# Gotchas baked in: CPU-only; NO USE_KINETO=0 (breaks profiler_kineto.cpp); BUILD_TEST=1
# (test files load libtorchbind_test.so); cmake>=4 needs CMAKE_POLICY_VERSION_MINIMUM.
export USE_CUDA=0 USE_DISTRIBUTED=0 USE_MKLDNN=1 USE_FBGEMM=1 BUILD_TEST=1 \
       MAX_JOBS="${MAX_JOBS:-64}" CMAKE_POLICY_VERSION_MINIMUM=3.5
export PATH="$ENV/bin:$PATH" LD_LIBRARY_PATH="/usr/lib64:$ENV/lib"
( cd "$SRC" && "$ENV/bin/python" setup.py develop )

echo "== done. Runtime notes: python needs LD_LIBRARY_PATH=/usr/lib64:$ENV/lib ;"
echo "   run git WITHOUT that LD_LIBRARY_PATH (host git may be linked against a custom libc). =="
