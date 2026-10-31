#!/bin/bash
# tokenizers-cpp's Rust C shim (libtokenizers_c.a) for prplOS 5.1 musl, from the pinned commit and the
# lock file in patches/musl/tokenizers-cpp, with cargo --locked.
# Usage: build-tokenizers.sh x86_64|aarch64
#   TOKENIZERS_SRC   a checkout of mlc-ai/tokenizers-cpp at c586c52 (the patch is applied if it is missing)
#   CARGO_TARGET_DIR output root (default <TOKENIZERS_SRC>/rust/target); the archive is
#                    <CARGO_TARGET_DIR>/<arch>-unknown-linux-musl/release/libtokenizers_c.a
#   CARGO            cargo to run (default ~/.cargo/bin/cargo); RUSTUP_TOOLCHAIN selects a rustup toolchain
#   CARGO_EXTRA      extra cargo arguments, e.g. the vendor-directory --config of an offline build
set -euo pipefail
ARCH=${1:?usage: $0 x86_64|aarch64}
R=${R:-$HOME/build-litert}
SRC=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
TK=${TOKENIZERS_SRC:?TOKENIZERS_SRC=<tokenizers-cpp checkout>}
WANT=c586c52f93f7b060753bd2388eb96a105cb7374d
case "$ARCH" in
x86_64)
  SD=$R/prplos-5.1-staging; T=$SD/toolchain-x86_64_gcc-13.3.0_musl/bin; X=x86_64-openwrt-linux-musl
  CPUFLAGS=""; RUST_CPU="" ;;
aarch64)
  SD=$R/prplos-5.1-staging-aarch64; T=$SD/toolchain-aarch64_cortex-a53_gcc-13.3.0_musl/bin
  X=aarch64-openwrt-linux-musl; CPUFLAGS="-march=armv8-a+crc -mtune=cortex-a53"
  RUST_CPU="-C target-cpu=cortex-a53" ;;
*) echo "usage: $0 x86_64|aarch64" >&2; exit 2 ;;
esac
export STAGING_DIR=$SD
RT=$ARCH-unknown-linux-musl
git -C "$TK" merge-base --is-ancestor "$WANT" HEAD || { echo "$TK is not $WANT (+ patches)" >&2; exit 1; }
if [ ! -f "$TK/rust/Cargo.lock" ]; then
  git -C "$TK" apply "$SRC"/patches/musl/tokenizers-cpp/*.patch
fi
CARGO=${CARGO:-$HOME/.cargo/bin/cargo}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$TK/rust/target}
V=${RT//-/_}; U=$(echo "$RT" | tr a-z- A-Z_)
# The cc crate (onig_sys's bundled Oniguruma) compiles C with these; rustc links with the musl gcc.
export CC_$V=$T/$X-gcc CXX_$V=$T/$X-g++ AR_$V=$T/$X-gcc-ar CFLAGS_$V="$CPUFLAGS"
export CARGO_TARGET_${U}_LINKER=$T/$X-gcc
[ -n "$RUST_CPU" ] && export CARGO_TARGET_${U}_RUSTFLAGS="$RUST_CPU"
echo "tokenizers-cpp $(git -C "$TK" rev-parse HEAD), $("$(dirname "$CARGO")/rustc" --version), target $RT"
cd "$TK/rust"
"$CARGO" build --release --locked --target "$RT" -j"${JOBS:-12}" ${CARGO_EXTRA:-}
ls -la "$CARGO_TARGET_DIR/$RT/release/libtokenizers_c.a"
