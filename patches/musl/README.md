# musl patches

`git format-patch` files for the third-party sources et-server's musl build compiles. Each directory
holds the patches for one upstream source, numbered in the order they apply, against the commit named
here. `tools/apply-musl-patches.sh <executorch-src> [<torch-include-dir>]` applies them idempotently;
`tools/et-musl-build.sh` runs it before configuring ExecuTorch.

| Directory | Upstream | Base commit | Applies to |
|---|---|---|---|
| `executorch/` | https://github.com/pytorch/executorch | `3b60683923245cf472b7323426920e15623ba361` (tag `v1.5.1`) | the checkout, `git apply` (or `git am`) at its root |
| `pytorch/` | https://github.com/pytorch/pytorch | `cf30153c4c131c8164ee7798e5022d810682e2cb` (the `git_version` of the torch 2.13.0 wheel) | a PyTorch checkout (`-p1`), or the wheel's `site-packages/torch/include` (`-p1` for `torch/headeronly/...`, `-p3` for `aten/src/ATen/...`) |

## executorch

- `0001-c10-declare-the-musl-__assert_fail-when-EXECUTORCH_L.patch`: ExecuTorch's vendored c10
  `Macros.h` declares the glibc `__assert_fail(..., unsigned int line, ...)`; musl's is `int line`, as
  Emscripten's (whose libc is musl), and GCC rejects the conflict. musl defines no identifying macro,
  so the Emscripten branch is keyed on `-DEXECUTORCH_LIBC_MUSL`, which the build passes in
  `CMAKE_C_FLAGS`/`CMAKE_CXX_FLAGS` and et-server's own `CMakeLists.txt` defines. Both architectures.

## pytorch

ExecuTorch's optimized CPU kernels (`kernels/optimized`, `cpublas`, `optimized_portable_kernels`)
include PyTorch headers (c10, ATen vectorization) from the torch package the build's Python finds
(`find_package_torch_headers` in `tools/cmake/Utils.cmake`), not from ExecuTorch's tree.

- `0001-headeronly-declare-the-musl-__assert_fail-when-EXECU.patch`: the same `__assert_fail` fix in
  PyTorch's copy of `torch/headeronly/macros/Macros.h`. Both architectures.
- `0002-vec128-spell-uint-as-unsigned-int-in-the-aarch64-uin.patch`: `vec128_uint_aarch64.h` uses the
  glibc/BSD typedef `uint`, which musl only declares in `<sys/types.h>` and which nothing in the include
  chain pulls in on musl: "'uint' has not been declared". aarch64 only (the header is not compiled on
  x86_64). Earlier builds force-included `sys/types.h` (`-include sys/types.h` in `CMAKE_CXX_FLAGS`)
  instead; the patch replaces that flag.

The torch wheel installs its headers with an extra `#if !defined(TORCH_STABLE_ONLY) ...` wrapper line
around `vec128_uint_aarch64.h`; the patch applies to both layouts with a one-line offset.
