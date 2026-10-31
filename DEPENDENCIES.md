# Dependencies

Every external source an et-server musl build reads, as the pinned trees resolve them, how to supply
each one for a build with no network, and its licence. Commits are from `git submodule status
--recursive` in the ExecuTorch v1.5.1 checkout and from the lock files, not from upstream docs.

The build has three stages: ExecuTorch's static libraries (`tools/et-musl-build.sh <arch>`), the
tokenizers C shim (Rust, `cargo build`), and et-server itself (`build-prplos-<arch>.sh`). The
ExecuTorch stage of the x86_64 build in this branch was run with no network (`unshare -rn`, a user and
network namespace with only a down loopback), as was the et-server stage given `HTTPLIB_SRC` and
`JSON_SRC`.

## Summary for a packager

- ExecuTorch: one git checkout of `v1.5.1` with its submodules (below), plus `patches/musl/executorch`.
  Nothing else is fetched at build time, except KleidiAI on aarch64 (one zip, give
  `KLEIDIAI_SOURCE_DIR`).
- PyTorch: the build needs PyTorch's c10/ATen headers and `torchgen` from the host Python (see
  "Host tools"). They come from a torch package, not from ExecuTorch's tree. `patches/musl/pytorch`
  applies to them.
- et-server: two CMake FetchContent sources (cpp-httplib, nlohmann/json); give
  `FETCHCONTENT_SOURCE_DIR_HTTPLIB` / `FETCHCONTENT_SOURCE_DIR_JSON` (the build scripts take
  `HTTPLIB_SRC` / `JSON_SRC`).
- Tokenizers shim: mlc-ai/tokenizers-cpp plus 81 Rust crates from `rust/Cargo.lock`
  (`cargo vendor`, then `cargo build --offline --locked`).
- Licences: all permissive (BSD, MIT, Apache-2.0, MPL-2.0 for Eigen, Unicode/Unlicense options among
  the crates). No GPL, LGPL or non-redistributable component is compiled or linked; nothing in the
  default build is CUDA, QAIRT or another vendor SDK. The source checkout carries more submodules than
  the build reads (Vulkan, MLX, Cadence, cutlass, buck2 prelude...); a feed may drop them from its
  download, but only those marked "not read" below.

## ExecuTorch and its submodules

`https://github.com/pytorch/executorch` at `3b60683923245cf472b7323426920e15623ba361` (tag `v1.5.1`,
2026-09-23). Patch: `patches/musl/executorch/0001-*.patch`.

"Used" is what the x86_64 build tree references: compiled or included (`ninja -t deps`), or only
configured (CMake reads its `CMakeLists.txt` because the `linux` preset turns the LLM extension on,
and the targets exist in `build.ninja`, but et-server's targets build none of them). A configured
submodule must be present for CMake to configure; a "not read" one can be absent.

| Path | URL | Commit | Used | Licence |
|---|---|---|---|---|
| `backends/xnnpack/third-party/XNNPACK` | https://github.com/google/XNNPACK.git | `92a7ad501b9516f9fecae119e0146dd1f58e54c1` | compiled, linked | BSD-3-Clause |
| `backends/xnnpack/third-party/cpuinfo` | https://github.com/pytorch/cpuinfo.git | `f9a03241f8c3d4ed0c9728f5d70bff873d43d4e0` | compiled, linked | BSD-2-Clause |
| `backends/xnnpack/third-party/pthreadpool` | https://github.com/google/pthreadpool.git | `a56dcd79c699366e7ac6466792c3025883ff7704` | compiled, linked | BSD-2-Clause |
| `backends/xnnpack/third-party/FXdiv` | https://github.com/Maratyszcza/FXdiv.git | `b408327ac2a15ec3e43352421954f5b1967701d1` | headers | MIT |
| `backends/xnnpack/third-party/FP16` | https://github.com/Maratyszcza/FP16.git | `4dfe081cf6bcd15db339cf2680b9281b8451eeb3` | not read by this build (no include or CMake reference; releases still ship its licence text) | MIT |
| `kernels/optimized/third-party/eigen` | https://gitlab.com/libeigen/eigen.git | `729443409942a1816ddf74b95224003b83f4925c` (3.4.90) | compiled (`eigen_blas`), linked | MPL-2.0 (files also under BSD/Apache/MINPACK) |
| `third-party/flatbuffers` | https://github.com/google/flatbuffers.git | `595bf0007ab1929570c7671f091313c8fc20644e` | headers; host `flatc` built from it | Apache-2.0 |
| `third-party/flatcc` | https://github.com/dvidelabs/flatcc.git | `896db54787e8b730a6be482c69324751f3f5f117` | configured (host `flatcc`, devtools only) | Apache-2.0 |
| `third-party/gflags` | https://github.com/gflags/gflags.git | `a738fdf9338412f83ab3f26f31ac11ed3f3ec4bd` | compiled for `executor_runner` only, not in et-server | BSD-3-Clause |
| `third-party/ao` | https://github.com/pytorch/ao.git | `5f2baf9d575cf732362594c998c399902942531f` | headers (optimized kernels) | BSD-3-Clause |
| `third-party/pocketfft` | https://github.com/mreineck/pocketfft | `8187407446316c3d16f15e5395dabd4b22f4fec7` | headers | BSD-3-Clause |
| `third-party/json` | https://github.com/nlohmann/json.git | `ac0133ea89a8fd730a9cc9649c5226a9995dc3e1` | configured | MIT |
| `extension/llm/tokenizers` | https://github.com/meta-pytorch/tokenizers.git | `61a10ac61245a68374ace4900e49cd2aba65fb39` | configured | BSD-3-Clause |
| `extension/llm/tokenizers/third-party/abseil-cpp` | https://github.com/abseil/abseil-cpp.git | `7d96b2e93d9a84530748b68abd2f97595c51ecf4` | configured | Apache-2.0 |
| `extension/llm/tokenizers/third-party/json` | https://github.com/nlohmann/json.git | `e6cafa573aac6ed9227f752a5371c0b3f436307d` | configured | MIT |
| `extension/llm/tokenizers/third-party/pcre2` | https://github.com/PCRE2Project/pcre2.git | `2e03e323339ab692640626f02f8d8d6f95bff9c6` | configured | BSD-3-Clause (PCRE2 licence) |
| `extension/llm/tokenizers/third-party/pcre2/deps/sljit` | https://github.com/zherczeg/sljit.git | `98802d939be3f0ccc2c538d611ec4c728564e8cf` | not read | BSD-2-Clause |
| `extension/llm/tokenizers/third-party/re2` | https://github.com/google/re2.git | `6dcd83d60f7944926bfd308cc13979fc53dd69ca` | configured | BSD-3-Clause |
| `extension/llm/tokenizers/third-party/sentencepiece` | https://github.com/google/sentencepiece.git | `bcc6390ab257c13056b7318983b35b8e1bbf442d` | configured | Apache-2.0 |
| `third-party/googletest` | https://github.com/google/googletest.git | `e2239ee6043f73722e7aa812a459f54a28552929` | not read (tests off) | BSD-3-Clause |
| `third-party/pybind11` | https://github.com/pybind/pybind11.git | `d03662f0984f652b60e7ddce53d3868002275197` | not read (pybind off) | BSD-3-Clause |
| `third-party/prelude` | https://github.com/facebook/buck2-prelude.git | `48c249f8c7b99ff501d6e857754760315072b306` | not read (Buck only) | Apache-2.0 / MIT |
| `shim` | https://github.com/facebook/buck2-shims-meta | `52196a3be467802f1cadce1debe7d8712e7327ee` | not read (Buck only) | Apache-2.0 / MIT |
| `third-party/ios-cmake` | https://github.com/leetal/ios-cmake | `06465b27698424cf4a04a5ca4904d50a3c966c45` | not read | BSD-3-Clause |
| `third-party/ao/third_party/cutlass` | https://github.com/NVIDIA/cutlass | `e51efbfe18fe4f4cbb66ab814c55bf4aa0185491` | not read (CUDA) | BSD-3-Clause |
| `backends/vulkan/third-party/Vulkan-Headers` | https://github.com/KhronosGroup/Vulkan-Headers | `8864cdc896bbc2a9b6eb36b3218fc9ef57908d77` | not read | Apache-2.0 / MIT |
| `backends/vulkan/third-party/VulkanMemoryAllocator` | https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git | `a6bfc237255a6bac1513f7c1ebde6d8aed6b5191` | not read | MIT |
| `backends/vulkan/third-party/volk` | https://github.com/zeux/volk | `49ba6858c13516019d699d94c31d5814025dd005` | not read | MIT |
| `backends/mlx/third-party/mlx` | https://github.com/ml-explore/mlx.git | `1f8e74e3f12f31365464a6867c6579f0e9b29d85` | not read (Apple) | MIT |
| `backends/cadence/utils/FACTO` | https://github.com/pytorch-labs/FACTO.git | `3b8c778c99766a8b4d0d04563ae0b16cbb276829` | not read | BSD-3-Clause |

Offline: clone with `git clone --recurse-submodules` (or fetch each commit above as a tarball into its
path) in the download step. CMake then needs nothing from the network on x86_64.

### Fetched by CMake at configure time (aarch64 only)

| Source | URL | Pin | Supply locally | Licence |
|---|---|---|---|---|
| KleidiAI (XNNPACK's Arm microkernels, `EXECUTORCH_XNNPACK_ENABLE_KLEIDI=ON`, the default) | https://gitlab.arm.com/kleidi/kleidiai/-/archive/b87ef9c94f45f11c81a6b1fdaed1b2b45ea58c0c/kleidiai-b87ef9c94f45f11c81a6b1fdaed1b2b45ea58c0c.zip | sha256 `b147799b94c51f5e57492930bfd9e5294fb7ffe44fee1dbcd3f8048adeedd5e3` (from XNNPACK's `cmake/DownloadKleidiAI.cmake`) | unzip it and pass `KLEIDIAI_SOURCE_DIR=<dir>` to `tools/et-musl-build.sh` (`-DKLEIDIAI_SOURCE_DIR`) | Apache-2.0 (some files BSD-3-Clause) |

On x86_64 XNNPACK disables KleidiAI and downloads nothing. XNNPACK's other download fallbacks
(cpuinfo, pthreadpool, FP16, FXdiv, googletest, benchmark) are not taken: ExecuTorch passes the
submodule paths, and tests and benchmarks are off.

## Host tools the ExecuTorch build runs

| Tool | Where it comes from | Offline |
|---|---|---|
| `flatc` (host) | built during the ExecuTorch build from `third-party/flatbuffers` (an `ExternalProject` with the host compiler: `flatc_ep`) | nothing to fetch. `FLATC_EXECUTABLE=<host flatc>` reuses a prebuilt one instead; it must come from the same flatbuffers commit (`595bf00`), since its generated headers are compiled against that tree's runtime headers. The `host-flatc` tree on ai4 (flatbuffers 25.9.23) is LiteRT's, not this one. |
| `flatcc` (host) | `third-party/flatcc` (`flatcc_ep`) | only for devtools/etdump; not built for et-server. `FLATCC_EXECUTABLE` overrides it. |
| Python 3 with `torch` (headers + `torchgen`), `pyyaml` | the build venv: torch 2.13.0 (`git_version` `cf30153c4c131c8164ee7798e5022d810682e2cb`); `PYTHON_EXECUTABLE` | ExecuTorch's CMake asks Python for the torch package path (`find_spec('torch')`) and includes `<torch>/include` (c10, ATen vectorization, `torch/headeronly`) in the optimized kernels, and runs `python -m codegen.gen` / `codegen.tools.gen_oplist` with `torchgen`'s `native_functions.yaml` and `tags.yaml` for the kernel registration. Supply a host Python with the torch wheel (`pip download torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu --no-deps`, about 200 MB; the CPU wheel is enough, only headers and Python files are used) and pyyaml, installed in the download step. Building PyTorch is not needed. |

Two cautions for a packager:
- ExecuTorch v1.5.1's `torch_pin.py` names torch 2.14.0; the builds here used 2.13.0 (the version the
  venv held), and the `patches/musl/pytorch` patches are cut against 2.13.0's commit. A feed that uses
  2.14.0 headers has to re-check both patches (and should expect `git apply` offsets).
- The torch wheel's include directory also carries a `pthreadpool.h` and a `cpuinfo.h`; the build
  includes them from there in a few translation units, next to the submodule copies. They agree for
  these pins; a different torch version could make them disagree.

The Python side does not run on the target and nothing from it is linked, except the c10/ATen header
code compiled into the optimized kernels (PyTorch's BSD-3-Clause licence, shipped as
`licenses/PyTorch-LICENSE`).

## PyTorch

`https://github.com/pytorch/pytorch` at `cf30153c4c131c8164ee7798e5022d810682e2cb` (release/2.13,
the torch 2.13.0 wheel's `git_version`). Headers only, through the torch package as above. Patches:
`patches/musl/pytorch/0001-*.patch` (both arches) and `0002-*.patch` (aarch64). Licence BSD-3-Clause.

## et-server's own fetches (CMake FetchContent)

| Name | URL | Commit | Supply locally | Licence |
|---|---|---|---|---|
| `httplib` | https://github.com/yhirose/cpp-httplib.git | `a7bc00e3307fecdb4d67545e93be7b88cfb1e186` (v0.18.3) | `-DFETCHCONTENT_SOURCE_DIR_HTTPLIB=<dir>` (`HTTPLIB_SRC`) | MIT |
| `json` | https://github.com/nlohmann/json.git | `9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03` (v3.11.3) | `-DFETCHCONTENT_SOURCE_DIR_JSON=<dir>` (`JSON_SRC`) | MIT |

Vendored: `third_party/stb/stb_image.h` (v2.30, commit `013ac3b`, public domain or MIT).

## Tokenizers C shim

| Source | URL | Pin | Supply locally | Licence |
|---|---|---|---|---|
| tokenizers-cpp | https://github.com/mlc-ai/tokenizers-cpp.git | `c586c52f93f7b060753bd2388eb96a105cb7374d` | git checkout; only `rust/` and `include/tokenizers_c.h` are used (its `msgpack` and `sentencepiece` submodules are not) | Apache-2.0 |
| 81 Rust crates (HuggingFace `tokenizers` 0.21.4, `onig_sys` 69.9.3 with bundled Oniguruma, ...) | crates.io | `rust/Cargo.lock` (checksums) | `cargo vendor` in the download step, then `cargo build --release --locked --offline --target <arch>-unknown-linux-musl` with a `.cargo/config.toml` pointing at the vendor directory | MIT / Apache-2.0 / BSD-2-Clause / Unlicense / Unicode options, per crate in `licenses/rust-crates.txt` of a release (`tools/rust_licenses.py`) |
| Rust toolchain | rustup stable with the `x86_64-unknown-linux-musl` / `aarch64-unknown-linux-musl` std | the toolchain the shim was built with | OpenWrt's rust host package, or a rustup tarball | MIT OR Apache-2.0 |

The shim archives used for the builds in this branch were the existing ones on ai4
(`tokenizers-cpp/rust/target/x86_64-unknown-linux-musl/release/libtokenizers_c.a`,
`aarch64/tokenizers-target/aarch64-unknown-linux-musl/release/libtokenizers_c.a`); they were not
rebuilt.

## Toolchain and target

prplOS 5.1 SDK: gcc 13.3.0, musl (`x86_64-openwrt-linux-musl`, `aarch64-openwrt-linux-musl` for
cortex-a53 with `-march=armv8-a+crc -mtune=cortex-a53`). The binary needs `libc.so`, `libgcc_s.so.1`
and `libstdc++.so.6` from the image; everything else is static.
