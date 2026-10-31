#!/bin/bash
# Build et-server musl-native for prplOS 5.1 aarch64 (cortex-a53) on the ai4 build host layout.
# Run it under a memory cap, e.g.
#   systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 ./build-prplos-aarch64.sh
# Offline: HTTPLIB_SRC and JSON_SRC name local checkouts of the two FetchContent sources
# (DEPENDENCIES.md); unset, CMake clones them.
set -euo pipefail
R=${R:-$HOME/build-litert}
SD=$R/prplos-5.1-staging-aarch64
T=$SD/toolchain-aarch64_cortex-a53_gcc-13.3.0_musl/bin
# -march/-mtune rather than -mcpu=cortex-a53: see tools/et-musl-build.sh.
CPUFLAGS="-march=armv8-a+crc -mtune=cortex-a53"
export STAGING_DIR=$SD
SRC=$(cd "$(dirname "$0")" && pwd)
BLD=${BLD:-$SRC/build-prplos-aarch64}
EXECUTORCH_SRC=${EXECUTORCH_SRC:-$R/et/executorch}
EXECUTORCH_BUILD=${EXECUTORCH_BUILD:-$R/aarch64/executorch-musl-aarch64}
JOBS=${JOBS:-6}
FC=()
[ -n "${HTTPLIB_SRC:-}" ] && FC+=(-DFETCHCONTENT_SOURCE_DIR_HTTPLIB="$HTTPLIB_SRC")
[ -n "${JSON_SRC:-}" ] && FC+=(-DFETCHCONTENT_SOURCE_DIR_JSON="$JSON_SRC")
# The curated tree's selected-operator archive when it has one (tools/et-musl-build.sh, the default),
# else every kernel (an ET_KERNELS=full tree).
KLIB=${ET_SERVER_KERNELS_LIB:-$EXECUTORCH_BUILD/libexecutorch_selected_kernels.a}
[ -f "$KLIB" ] || KLIB=$EXECUTORCH_BUILD/configurations/liboptimized_native_cpu_ops_lib.a
echo "operator registration: $KLIB"
rm -rf "$BLD"
cmake -S "$SRC" -B "$BLD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=$T/aarch64-openwrt-linux-musl-gcc -DCMAKE_CXX_COMPILER=$T/aarch64-openwrt-linux-musl-g++ \
  -DCMAKE_C_FLAGS="$CPUFLAGS" -DCMAKE_CXX_FLAGS="$CPUFLAGS" \
  -DCMAKE_SYSROOT=$SD/target-aarch64_cortex-a53_musl "${FC[@]}" \
  -DEXECUTORCH_SRC="$EXECUTORCH_SRC" -DEXECUTORCH_BUILD="$EXECUTORCH_BUILD" \
  -DET_SERVER_KERNELS_LIB="$KLIB" \
  -DTOKENIZERS_CPP_SRC=$R/tokenizers-cpp \
  -DTOKENIZERS_C_LIB=${TOKENIZERS_C_LIB:-$R/aarch64/tokenizers-target/aarch64-unknown-linux-musl/release/libtokenizers_c.a}
nice -n 19 cmake --build "$BLD" -j"$JOBS"
"$T/aarch64-openwrt-linux-musl-strip" -o "$BLD/et-server.stripped" "$BLD/et-server"
file "$BLD/et-server"
ls -la "$BLD/et-server" "$BLD/et-server.stripped" | awk '{print $5, $9}'
NEEDED=$("$T/aarch64-openwrt-linux-musl-readelf" -d "$BLD/et-server" | grep NEEDED | grep -o '\[[^]]*\]' | tr -d '[]' | sort | tr '\n' ' ')
echo "NEEDED: $NEEDED"
if [ "$NEEDED" != "libc.so libgcc_s.so.1 libstdc++.so.6 " ]; then
  echo "unexpected NEEDED set (want libc.so libgcc_s.so.1 libstdc++.so.6)" >&2
  exit 1
fi
