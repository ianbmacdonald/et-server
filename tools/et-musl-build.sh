#!/bin/bash
# Build the ExecuTorch v1.5.1 static libraries et-server links (XNNPACK delegate, optimized CPU
# kernels, module/tensor/data-loader extensions) for prplOS 5.1 x86_64 musl, on the ai4 layout.
#
#   tools/et-musl-build.sh            # re-runs itself under systemd-run with MemoryMax=${MEMMAX:-24G}
#
# Inputs: $R/et/executorch (a v1.5.1 checkout; the directory must be named "executorch", its
# headers are included as <executorch/...>), the prplOS 5.1 toolchain and target staging under
# $R/prplos-5.1-staging, and a Python venv with executorch 1.5.1 / torch at $R/tflite-venv.
# Output: the raw CMake build tree $R/executorch-musl-x86_64 (no install step), with its configure
# and build logs next to it as <tree>.cfg.log and <tree>.build.log.
set -euo pipefail
if [ -z "${ET_BUILD_SCOPED:-}" ]; then
  exec systemd-run --user --scope -q -p MemoryMax="${MEMMAX:-24G}" -p MemorySwapMax=0 \
    env ET_BUILD_SCOPED=1 "$0" "$@"
fi
R=${R:-$HOME/build-litert}
SD=$R/prplos-5.1-staging
T=$SD/toolchain-x86_64_gcc-13.3.0_musl/bin
S=$SD/target-x86_64_musl
export STAGING_DIR=$SD
SRC=${EXECUTORCH_SRC:-$R/et/executorch}
B=${EXECUTORCH_BUILD:-$R/executorch-musl-x86_64}
PY=${PY:-$R/tflite-venv/bin/python}
JOBS=${JOBS:-6}
WEIGHT_CACHE=${WEIGHT_CACHE:-OFF}
# Selective build (ExecuTorch's own mechanism): ET_SELECT_OPS_MODEL=<model.pte> registers only the
# operators that model's non-delegated graph calls; ET_SELECT_OPS_LIST="aten::add.out,..." an explicit
# list. Either adds libexecutorch_selected_kernels.a to the tree; build et-server with
# ET_SERVER_KERNELS_LIB=$B/libexecutorch_selected_kernels.a. Unset: every optimized+portable kernel.
SELECT=()
[ -n "${ET_SELECT_OPS_MODEL:-}" ] && SELECT+=(-DEXECUTORCH_SELECT_OPS_MODEL="$(readlink -f "$ET_SELECT_OPS_MODEL")")
[ -n "${ET_SELECT_OPS_LIST:-}" ] && SELECT+=(-DEXECUTORCH_SELECT_OPS_LIST="$ET_SELECT_OPS_LIST")
[ ${#SELECT[@]} -le 1 ] || { echo "set ET_SELECT_OPS_MODEL or ET_SELECT_OPS_LIST, not both" >&2; exit 1; }
WANT_TAG=v1.5.1

[ "$(basename "$SRC")" = executorch ] || { echo "the checkout directory must be named 'executorch': $SRC" >&2; exit 1; }
head=$(git -C "$SRC" rev-parse HEAD)
tag=$(git -C "$SRC" rev-parse "$WANT_TAG^{commit}")
[ "$head" = "$tag" ] || { echo "$SRC HEAD $head is not $WANT_TAG ($tag)" >&2; exit 1; }

# musl declares __assert_fail(..., int line, ...) like Emscripten (whose libc IS musl); ExecuTorch's
# vendored c10 header otherwise declares the glibc form and GCC rejects the conflict. musl defines no
# identifying macro, so key the Emscripten branch on a flag we pass. Idempotent.
patch_macros() {
  grep -q EXECUTORCH_LIBC_MUSL "$1" ||
    sed -i 's|#elif (defined(__EMSCRIPTEN__))|#elif (defined(__EMSCRIPTEN__) \|\| defined(EXECUTORCH_LIBC_MUSL))|' "$1"
  grep -q "defined(EXECUTORCH_LIBC_MUSL)" "$1"
}
patch_macros "$SRC/runtime/core/portable_type/c10/torch/headeronly/macros/Macros.h"
# The optimized kernels also include PyTorch's copy of the same header from the venv's torch wheel.
patch_macros "$(ls "$R"/tflite-venv/lib/python3*/site-packages/torch/include/torch/headeronly/macros/Macros.h)"

rm -rf "$B"
cmake -S "$SRC" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  --preset linux \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=x86_64 \
  -DCMAKE_C_COMPILER=$T/x86_64-openwrt-linux-musl-gcc -DCMAKE_CXX_COMPILER=$T/x86_64-openwrt-linux-musl-g++ \
  -DCMAKE_C_FLAGS="-D_GNU_SOURCE -DEXECUTORCH_LIBC_MUSL" -DCMAKE_CXX_FLAGS="-D_GNU_SOURCE -DEXECUTORCH_LIBC_MUSL" \
  -DCMAKE_SYSROOT=$S -DCMAKE_FIND_ROOT_PATH=$S \
  -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
  -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DPYTHON_EXECUTABLE=$PY \
  -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_XNNPACK_ENABLE_WEIGHT_CACHE="$WEIGHT_CACHE" \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_PYBIND=OFF -DEXECUTORCH_BUILD_TESTS=OFF "${SELECT[@]}" > "$B.cfg.log" 2>&1
# executor_runner pulls in the core, kernels and XNNPACK; the extension libraries et-server also
# links are separate targets.
TARGETS=(executor_runner)
[ ${#SELECT[@]} -eq 1 ] && TARGETS+=(executorch_selected_kernels)
nice -n 19 cmake --build "$B" -j"$JOBS" --target "${TARGETS[@]}" extension_module_static \
  extension_data_loader extension_flat_tensor extension_named_data_map extension_tensor \
  extension_threadpool > "$B.build.log" 2>&1
RUN=$(find "$B" -name executor_runner -type f -perm -u+x | head -1)
"$T/x86_64-openwrt-linux-musl-strip" -o "$B/executor_runner.stripped" "$RUN"
file "$RUN"
echo "ExecuTorch $(git -C "$SRC" describe --tags) $head built in $B (weight cache default $WEIGHT_CACHE)"
