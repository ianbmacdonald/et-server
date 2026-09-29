#!/bin/bash
# Build et-server musl-native for prplOS 5.1 x86_64 on the ai4 build host layout.
# Run it under a memory cap, e.g.
#   systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 ./build-prplos-x86_64.sh
set -euo pipefail
R=${R:-$HOME/build-litert}
SD=$R/prplos-5.1-staging
T=$SD/toolchain-x86_64_gcc-13.3.0_musl/bin
export STAGING_DIR=$SD
SRC=$(cd "$(dirname "$0")" && pwd)
BLD=${BLD:-$SRC/build-prplos-x86_64}
EXECUTORCH_SRC=${EXECUTORCH_SRC:-$R/et/executorch}
EXECUTORCH_BUILD=${EXECUTORCH_BUILD:-$R/executorch-musl-x86_64}
JOBS=${JOBS:-6}
rm -rf "$BLD"
cmake -S "$SRC" -B "$BLD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=x86_64 \
  -DCMAKE_C_COMPILER=$T/x86_64-openwrt-linux-musl-gcc -DCMAKE_CXX_COMPILER=$T/x86_64-openwrt-linux-musl-g++ \
  -DCMAKE_SYSROOT=$SD/target-x86_64_musl \
  -DEXECUTORCH_SRC="$EXECUTORCH_SRC" -DEXECUTORCH_BUILD="$EXECUTORCH_BUILD" \
  -DTOKENIZERS_CPP_SRC=$R/tokenizers-cpp \
  -DTOKENIZERS_C_LIB=$R/tokenizers-cpp/rust/target/x86_64-unknown-linux-musl/release/libtokenizers_c.a
nice -n 19 cmake --build "$BLD" -j"$JOBS"
"$T/x86_64-openwrt-linux-musl-strip" -o "$BLD/et-server.stripped" "$BLD/et-server"
file "$BLD/et-server"
ls -la "$BLD/et-server" "$BLD/et-server.stripped" | awk '{print $5, $9}'
NEEDED=$("$T/x86_64-openwrt-linux-musl-readelf" -d "$BLD/et-server" | grep NEEDED | grep -o '\[[^]]*\]' | tr -d '[]' | sort | tr '\n' ' ')
echo "NEEDED: $NEEDED"
if [ "$NEEDED" != "libc.so libgcc_s.so.1 libstdc++.so.6 " ]; then
  echo "unexpected NEEDED set (want libc.so libgcc_s.so.1 libstdc++.so.6)" >&2
  exit 1
fi
