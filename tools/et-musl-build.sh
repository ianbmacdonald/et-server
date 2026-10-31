#!/bin/bash
# Build the ExecuTorch v1.5.1 static libraries et-server links (XNNPACK delegate, optimized CPU
# kernels, module/tensor/data-loader extensions) for prplOS 5.1 musl, x86_64 or aarch64 (cortex-a53),
# on the ai4 layout.
#
#   tools/et-musl-build.sh [x86_64|aarch64]   # re-runs itself under systemd-run with MemoryMax=${MEMMAX:-24G}
#                                             # (ET_BUILD_SCOPED=1: already capped, run in place)
#
# Inputs: $R/et/executorch (a v1.5.1 checkout; the directory must be named "executorch", its
# headers are included as <executorch/...>), the prplOS 5.1 toolchain and target staging under
# $R/prplos-5.1-staging[-aarch64], and a Python venv with torch 2.13.0 at $R/tflite-venv (its headers
# and torchgen; see DEPENDENCIES.md). patches/musl is applied to both (tools/apply-musl-patches.sh).
# Output: the raw CMake build tree $R/executorch-musl-x86_64 or $R/aarch64/executorch-musl-aarch64 (no install step), with its configure
# and build logs next to it as <tree>.cfg.log and <tree>.build.log.
# Offline: every source is in the checkout's submodules except KleidiAI (aarch64), which XNNPACK
# downloads unless KLEIDIAI_SOURCE_DIR names an unpacked copy. FLATC_EXECUTABLE (optional) is a
# prebuilt host flatc from the same flatbuffers commit instead of building it from third-party/flatbuffers.
set -euo pipefail
if [ -z "${ET_BUILD_SCOPED:-}" ]; then
  exec systemd-run --user --scope -q -p MemoryMax="${MEMMAX:-24G}" -p MemorySwapMax=0 \
    env ET_BUILD_SCOPED=1 "$0" "$@"
fi
ARCH=${1:-x86_64}
R=${R:-$HOME/build-litert}
case "$ARCH" in
x86_64)
  SD=$R/prplos-5.1-staging; T=$SD/toolchain-x86_64_gcc-13.3.0_musl/bin; S=$SD/target-x86_64_musl
  X=x86_64-openwrt-linux-musl; CPUFLAGS="" ;;
aarch64)
  SD=$R/prplos-5.1-staging-aarch64; T=$SD/toolchain-aarch64_cortex-a53_gcc-13.3.0_musl/bin
  S=$SD/target-aarch64_cortex-a53_musl; X=aarch64-openwrt-linux-musl
  # -march/-mtune, not -mcpu=cortex-a53: a global -mcpu overrides the per-file -march=armv8.2-a+...
  # XNNPACK and KleidiAI put on their runtime-dispatched kernels, and gas then rejects them.
  CPUFLAGS="-march=armv8-a+crc -mtune=cortex-a53" ;;
*) echo "usage: $0 [x86_64|aarch64]" >&2; exit 2 ;;
esac
export STAGING_DIR=$SD
SRC=${EXECUTORCH_SRC:-$R/et/executorch}
B=${EXECUTORCH_BUILD:-$R/executorch-musl-x86_64}
[ "$ARCH" = aarch64 ] && B=${EXECUTORCH_BUILD:-$R/aarch64/executorch-musl-aarch64}
PY=${PY:-$R/tflite-venv/bin/python}
JOBS=${JOBS:-6}
WEIGHT_CACHE=${WEIGHT_CACHE:-OFF}
# Operators (ExecuTorch's selective build). ET_KERNELS=curated (the default) registers only the operators
# in ops/curated.txt, through libexecutorch_selected_kernels.a (optimized kernels where ExecuTorch has one,
# else portable); build-prplos-<arch>.sh links that archive when the tree has it. ET_KERNELS=full: every
# optimized+portable kernel (configurations/liboptimized_native_cpu_ops_lib.a, the only one in the tree).
# Other selections: ET_SELECT_OPS_FILE=<file> (one operator per line), ET_SELECT_OPS_LIST="aten::add.out,..."
# or ET_SELECT_OPS_MODEL=<model.pte> (the operators that model's non-delegated graph calls).
case "${ET_KERNELS:-curated}" in
curated)
  if [ -z "${ET_SELECT_OPS_LIST:-}${ET_SELECT_OPS_MODEL:-}" ]; then
    ET_SELECT_OPS_FILE=${ET_SELECT_OPS_FILE:-$(dirname "$(readlink -f "$0")")/../ops/curated.txt}
  fi ;;
full)
  [ -z "${ET_SELECT_OPS_FILE:-}${ET_SELECT_OPS_LIST:-}${ET_SELECT_OPS_MODEL:-}" ] ||
    { echo "ET_KERNELS=full takes no ET_SELECT_OPS_*" >&2; exit 1; } ;;
*) echo "ET_KERNELS must be curated or full" >&2; exit 1 ;;
esac
if [ -n "${ET_SELECT_OPS_FILE:-}" ]; then
  [ -z "${ET_SELECT_OPS_LIST:-}" ] || { echo "set ET_SELECT_OPS_FILE or ET_SELECT_OPS_LIST, not both" >&2; exit 1; }
  ET_SELECT_OPS_LIST=$(grep -v '^#' "$ET_SELECT_OPS_FILE" | tr -d ' \t' | grep . | paste -sd, -)
fi
SELECT=()
[ -n "${ET_SELECT_OPS_MODEL:-}" ] && SELECT+=(-DEXECUTORCH_SELECT_OPS_MODEL="$(readlink -f "$ET_SELECT_OPS_MODEL")")
[ -n "${ET_SELECT_OPS_LIST:-}" ] && SELECT+=(-DEXECUTORCH_SELECT_OPS_LIST="$ET_SELECT_OPS_LIST")
[ ${#SELECT[@]} -le 1 ] || { echo "set ET_SELECT_OPS_MODEL or ET_SELECT_OPS_LIST, not both" >&2; exit 1; }
WANT_TAG=v1.5.1

[ "$(basename "$SRC")" = executorch ] || { echo "the checkout directory must be named 'executorch': $SRC" >&2; exit 1; }
head=$(git -C "$SRC" rev-parse HEAD)
tag=$(git -C "$SRC" rev-parse "$WANT_TAG^{commit}")
[ "$head" = "$tag" ] || { echo "$SRC HEAD $head is not $WANT_TAG ($tag)" >&2; exit 1; }

# patches/musl: musl's __assert_fail in both copies of c10's Macros.h (ExecuTorch's vendored one and the
# torch package's, which the optimized kernels include), and the aarch64 uint vectors' bare `uint`.
TORCH_INCLUDE=$("$PY" -c 'import importlib.util; print(importlib.util.find_spec("torch").submodule_search_locations[0])')/include
"$(dirname "$(readlink -f "$0")")/apply-musl-patches.sh" "$SRC" "$TORCH_INCLUDE"
EXTRA=()
[ -n "${KLEIDIAI_SOURCE_DIR:-}" ] && EXTRA+=(-DKLEIDIAI_SOURCE_DIR="$KLEIDIAI_SOURCE_DIR")
[ -n "${FLATC_EXECUTABLE:-}" ] && EXTRA+=(-DFLATC_EXECUTABLE="$FLATC_EXECUTABLE")

rm -rf "$B"
cmake -S "$SRC" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  --preset linux \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR="$ARCH" \
  -DCMAKE_C_COMPILER=$T/$X-gcc -DCMAKE_CXX_COMPILER=$T/$X-g++ \
  -DCMAKE_C_FLAGS="$CPUFLAGS -D_GNU_SOURCE -DEXECUTORCH_LIBC_MUSL" \
  -DCMAKE_CXX_FLAGS="$CPUFLAGS -D_GNU_SOURCE -DEXECUTORCH_LIBC_MUSL" \
  -DCMAKE_SYSROOT=$S -DCMAKE_FIND_ROOT_PATH=$S \
  -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
  -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DPYTHON_EXECUTABLE=$PY \
  -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_XNNPACK_ENABLE_WEIGHT_CACHE="$WEIGHT_CACHE" \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_PYBIND=OFF -DEXECUTORCH_BUILD_TESTS=OFF "${SELECT[@]}" "${EXTRA[@]}" > "$B.cfg.log" 2>&1
# executor_runner pulls in the core, kernels and XNNPACK; the extension libraries et-server also
# links are separate targets.
TARGETS=(executor_runner)
[ ${#SELECT[@]} -eq 1 ] && TARGETS+=(executorch_selected_kernels)
nice -n 19 cmake --build "$B" -j"$JOBS" --target "${TARGETS[@]}" extension_module_static \
  extension_data_loader extension_flat_tensor extension_named_data_map extension_tensor \
  extension_threadpool > "$B.build.log" 2>&1
RUN=$(find "$B" -name executor_runner -type f -perm -u+x | head -1)
"$T/$X-strip" -o "$B/executor_runner.stripped" "$RUN"
file "$RUN"
echo "ExecuTorch $(git -C "$SRC" describe --tags) $head built for $ARCH in $B (weight cache default $WEIGHT_CACHE)"
