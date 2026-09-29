#!/bin/bash
# Package an et-server release tarball from build-prplos-x86_64 (run on the build host).
# Usage: package-release.sh <version>
set -euo pipefail
VER=$1
R=${R:-$HOME/build-litert}
SRC=$(cd "$(dirname "$0")" && pwd)
BLD=${BLD:-$SRC/build-prplos-x86_64}
ET=${EXECUTORCH_SRC:-$R/et/executorch}
ETB=${EXECUTORCH_BUILD:-$R/executorch-musl-x86_64}
NAME=et-server-musl-x86_64-$VER
OUT=${OUT:-$R/bundle}
B=$OUT/$NAME
T=$R/prplos-5.1-staging/toolchain-x86_64_gcc-13.3.0_musl/bin
export STAGING_DIR=$R/prplos-5.1-staging

rm -rf "$B" && mkdir -p "$B/bin" "$B/licenses"
cp "$BLD/et-server.stripped" "$B/bin/et-server"
cp "$SRC/LICENSE" "$SRC/NOTICE" "$SRC/README.md" "$B/"
cp "$ET/LICENSE" "$B/licenses/ExecuTorch-LICENSE"
cp "$ET/backends/xnnpack/third-party/XNNPACK/LICENSE" "$B/licenses/XNNPACK-LICENSE"
cp "$ET/backends/xnnpack/third-party/cpuinfo/LICENSE" "$B/licenses/cpuinfo-LICENSE"
cp "$ET/backends/xnnpack/third-party/pthreadpool/LICENSE" "$B/licenses/pthreadpool-LICENSE"
cp "$ET/kernels/optimized/third-party/eigen/COPYING.MPL2" "$B/licenses/Eigen-COPYING.MPL2"
cp "$ET/third-party/flatbuffers/LICENSE" "$B/licenses/flatbuffers-LICENSE"
cp "$R/tokenizers-cpp/LICENSE" "$B/licenses/tokenizers-cpp-LICENSE"
cp "$BLD/_deps/httplib-src/LICENSE" "$B/licenses/cpp-httplib-LICENSE"
cp "$BLD/_deps/json-src/LICENSE.MIT" "$B/licenses/nlohmann-json-LICENSE"
ONIG=$(find "$HOME/.cargo/registry/src" -maxdepth 4 -path '*onig_sys*' -name COPYING | head -1)
[ -n "$ONIG" ] && cp "$ONIG" "$B/licenses/oniguruma-COPYING"
( cd "$R/tokenizers-cpp/rust" && "$HOME/.cargo/bin/cargo" metadata --format-version 1 \
    --filter-platform x86_64-unknown-linux-musl 2>/dev/null ) | python3 -c '
import json, sys
d = json.load(sys.stdin)
res = {n["id"] for n in d["resolve"]["nodes"]}
print("Rust crates statically linked into bin/et-server (via the HuggingFace tokenizers C shim):")
for name, ver, lic in sorted({(p["name"], p["version"], p.get("license") or "see crate") for p in d["packages"] if p["id"] in res}):
    print(f"  {name} {ver}  {lic}")
' > "$B/licenses/rust-crates.txt"
{
  echo "et-server $VER, musl-native for prplOS 5.1 x86_64 (gcc 13.3.0 musl toolchain)."
  echo "Source: https://github.com/ianbmacdonald/et-server @ $(git -C "$SRC" rev-parse --short HEAD)"
  echo "ExecuTorch: $(git -C "$ET" describe --tags) $(git -C "$ET" rev-parse HEAD), static, from $(basename "$ETB")."
  echo "ExecuTorch CMake flags:"
  grep -E '^(EXECUTORCH_BUILD_(XNNPACK|KERNELS_OPTIMIZED|EXTENSION_[A-Z_]+)|EXECUTORCH_XNNPACK_[A-Z_]+|CMAKE_BUILD_TYPE|CMAKE_CXX_FLAGS):' \
    "$ETB/CMakeCache.txt" | sed 's/^/  /'
  echo "Runtime libraries (from the prplOS image, not bundled):"
  "$T/x86_64-openwrt-linux-musl-readelf" -d "$B/bin/et-server" | grep -o 'NEEDED.*\[.*\]' | grep -o '\[[^]]*\]' | sed 's/^/  /'
} > "$B/BUILD-INFO.txt"

cd "$OUT"
tar czf "$NAME.tar.gz" "$NAME"
sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256"
echo "$NAME.tar.gz $(stat -c %s "$NAME.tar.gz") bytes sha256 $(cut -c1-64 "$NAME.tar.gz.sha256")"
cat "$B/BUILD-INFO.txt"
ls "$B/licenses"
