#!/bin/bash
# Package an et-server release tarball (run on the build host).
# Usage: package-release.sh <version> [x86_64|aarch64]
#   BLD               build tree (default build-prplos-<arch>)
#   EXECUTORCH_SRC    the ExecuTorch v1.5.1 checkout
#   EXECUTORCH_BUILD  the ExecuTorch tree the server was linked against
#   OUT               output directory (default $R/bundle)
set -euo pipefail
VER=$1
ARCH=${2:-x86_64}
R=${R:-$HOME/build-litert}
SRC=$(cd "$(dirname "$0")" && pwd)
case "$ARCH" in
x86_64)
  SD=$R/prplos-5.1-staging; T=$SD/toolchain-x86_64_gcc-13.3.0_musl/bin/x86_64-openwrt-linux-musl-
  ET=${EXECUTORCH_SRC:-$R/et/executorch}; ETB=${EXECUTORCH_BUILD:-$R/executorch-musl-x86_64}
  TOKDIR=$R/tokenizers-cpp/rust/target/x86_64-unknown-linux-musl/release
  HOST="prplOS 5.1 x86_64 (gcc 13.3.0 musl toolchain)" ;;
aarch64)
  SD=$R/prplos-5.1-staging-aarch64; T=$SD/toolchain-aarch64_cortex-a53_gcc-13.3.0_musl/bin/aarch64-openwrt-linux-musl-
  ET=${EXECUTORCH_SRC:-$R/aarch64/et/executorch}; ETB=${EXECUTORCH_BUILD:-$R/aarch64/executorch-musl-aarch64}
  TOKDIR=$R/aarch64/tokenizers-target/aarch64-unknown-linux-musl/release
  HOST="prplOS 5.1 aarch64 cortex-a53 (gcc 13.3.0 musl toolchain, -march=armv8-a+crc -mtune=cortex-a53)" ;;
*) echo "usage: $0 <version> [x86_64|aarch64]" >&2; exit 2 ;;
esac
export STAGING_DIR=$SD
BLD=${BLD:-$SRC/build-prplos-$ARCH}
NAME=et-server-musl-$ARCH-$VER
OUT=${OUT:-$R/bundle}
B=$OUT/$NAME
L=$B/licenses
need() {  # need <src> <dest-name>: licence texts are not optional
  [ -f "$1" ] || { echo "missing licence text $1" >&2; exit 1; }
  cp "$1" "$L/$2"
}

rm -rf "$B" && mkdir -p "$B/bin" "$L"
cp "$BLD/et-server.stripped" "$B/bin/et-server"
cp "$SRC/LICENSE" "$SRC/NOTICE" "$SRC/README.md" "$B/"
need "$ET/LICENSE" ExecuTorch-LICENSE
need "$ET/backends/xnnpack/third-party/XNNPACK/LICENSE" XNNPACK-LICENSE
need "$ET/backends/xnnpack/third-party/cpuinfo/LICENSE" cpuinfo-LICENSE
need "$ET/backends/xnnpack/third-party/pthreadpool/LICENSE" pthreadpool-LICENSE
need "$ET/backends/xnnpack/third-party/FP16/LICENSE" FP16-LICENSE
need "$ET/backends/xnnpack/third-party/FXdiv/LICENSE" FXdiv-LICENSE
need "$ET/kernels/optimized/third-party/eigen/COPYING.MPL2" Eigen-COPYING.MPL2
need "$ET/third-party/flatbuffers/LICENSE" flatbuffers-LICENSE
# runtime/core/portable_type/c10 and the optimized kernels' ATen vectorization headers (from the venv's torch
# wheel, which the ExecuTorch build includes) are PyTorch code.
need "$(ls "$R"/tflite-venv/lib/python3*/site-packages/torch-*.dist-info/licenses/LICENSE | head -1)" PyTorch-LICENSE
if [ "$ARCH" = aarch64 ]; then
  # The XNNPACK delegate links KleidiAI on aarch64 (nothing on x86_64).
  need "$ETB/kleidiai-source/LICENSES/Apache-2.0.txt" KleidiAI-LICENSE
  need "$ETB/kleidiai-source/LICENSES/BSD-3-Clause.txt" KleidiAI-BSD-3-Clause.txt
fi
need "$R/tokenizers-cpp/LICENSE" tokenizers-cpp-LICENSE
need "$BLD/_deps/httplib-src/LICENSE" cpp-httplib-LICENSE
need "$BLD/_deps/json-src/LICENSE.MIT" nlohmann-json-LICENSE
need "$SRC/third_party/stb/LICENSE" stb-LICENSE
# The Rust crates of the tokenizer shim (incl. HuggingFace tokenizers, onig, Oniguruma), per target.
python3 "$SRC/tools/rust_licenses.py" "$R/tokenizers-cpp/rust" "$ARCH-unknown-linux-musl" "$L" et-server \
  --cargo "$HOME/.cargo/bin/cargo"
RUSTC=$(strings "$TOKDIR/libtokenizers_c.a" | grep -o -m1 'rustc version [0-9][^ )]*' || true)
RD=$(ls -d "$HOME"/.rustup/toolchains/stable-*/share/doc/rust/licenses | head -1)
mkdir -p "$L/rust-std"
for f in Apache-2.0 MIT LLVM-exception Unicode-3.0; do need "$RD/$f.txt" "rust-std/$f.txt"; done
echo "The Rust standard library (${RUSTC:-rustc}) linked into the tokenizer shim: MIT OR Apache-2.0; its bundled libunwind: Apache-2.0 WITH LLVM-exception; Unicode data: Unicode-3.0." > "$L/rust-std/README"
KLIB=$(grep -E '^ET_SERVER_KERNELS_LIB:' "$BLD/CMakeCache.txt" | cut -d= -f2-)
{
  echo "et-server $VER, musl-native for $HOST."
  echo "Source: https://github.com/ianbmacdonald/et-server @ $(git -C "$SRC" rev-parse --short HEAD)"
  echo "ExecuTorch: $(git -C "$ET" describe --tags) $(git -C "$ET" rev-parse HEAD), static, from $(basename "$ETB")."
  if [ "$(basename "$KLIB")" = libexecutorch_selected_kernels.a ]; then
    SEL=$(grep -E '^EXECUTORCH_SELECT_OPS_LIST:' "$ETB/CMakeCache.txt" | cut -d= -f2-)
    echo "Operators: curated, ops/curated.txt ($(echo "$SEL" | tr , '\n' | grep -c .) operators outside the XNNPACK delegate: the union over"
    echo "  the six use cases of the gateway study - text classification, sentence embeddings, image classification,"
    echo "  object detection, audio, time series; 16 models incl. DistilBERT and MobileNetV2). Optimized kernels where"
    echo "  ExecuTorch has one, else portable. A model with another operator fails to load. Linking"
    echo "  configurations/liboptimized_native_cpu_ops_lib.a instead is the general build (every kernel)."
    echo "$SEL" | tr , ' ' | fold -s -w 100 | sed 's/^/  /'
  else
    echo "Operators: general build, every optimized and portable kernel ($(basename "$KLIB"))."
  fi
  echo "ExecuTorch CMake flags:"
  grep -E '^(EXECUTORCH_BUILD_(XNNPACK|KERNELS_OPTIMIZED|EXTENSION_[A-Z_]+)|EXECUTORCH_XNNPACK_[A-Z_]+|CMAKE_BUILD_TYPE|CMAKE_CXX_FLAGS):' \
    "$ETB/CMakeCache.txt" | sed 's/^/  /'
  [ "$ARCH" = aarch64 ] && echo "aarch64: also links $(basename "$ETB")/kleidiai/libkleidiai.a (XNNPACK KleidiAI kernels; CMAKE_CXX_STANDARD_LIBRARIES)."
  echo "Runtime libraries (from the prplOS image, not bundled):"
  "${T}readelf" -d "$B/bin/et-server" | grep -o 'NEEDED.*\[.*\]' | grep -o '\[[^]]*\]' | sed 's/^/  /'
} > "$B/BUILD-INFO.txt"
chmod -R go-w "$B"

cd "$OUT"
tar czf "$NAME.tar.gz" "$NAME"
sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256"
echo "$NAME.tar.gz $(stat -c %s "$NAME.tar.gz") bytes sha256 $(cut -c1-64 "$NAME.tar.gz.sha256")"
cat "$B/BUILD-INFO.txt"
ls "$L"
