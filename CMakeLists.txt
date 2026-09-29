cmake_minimum_required(VERSION 3.24)
project(et-server CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ExecuTorch is consumed as a raw CMake build tree (static archives, no install),
# e.g. built with tools/et-musl-build.sh, plus its source checkout for headers.
set(EXECUTORCH_SRC "" CACHE PATH "ExecuTorch v1.5.1 source checkout (the directory named 'executorch')")
set(EXECUTORCH_BUILD "" CACHE PATH "ExecuTorch CMake build tree configured from EXECUTORCH_SRC")
set(TOKENIZERS_CPP_SRC "" CACHE PATH "mlc-ai/tokenizers-cpp checkout (for include/tokenizers_c.h)")
set(TOKENIZERS_C_LIB "" CACHE FILEPATH "libtokenizers_c.a built for the target")
foreach(v EXECUTORCH_SRC EXECUTORCH_BUILD TOKENIZERS_CPP_SRC TOKENIZERS_C_LIB)
  if(NOT ${v})
    message(FATAL_ERROR "${v} is required")
  endif()
endforeach()

include(FetchContent)
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG a7bc00e3307fecdb4d67545e93be7b88cfb1e186)  # v0.18.3
FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG 9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03)  # v3.11.3
FetchContent_MakeAvailable(httplib json)

set(B "${EXECUTORCH_BUILD}")
# Archives whose objects register themselves from static initializers (the prim
# ops, the XNNPACK backend, the optimized CPU kernels): nothing references them
# by symbol, so without --whole-archive the linker drops them and load_method
# fails with a missing operator or backend. The set and the order below follow
# `ninja -t commands executor_runner` in that tree, minus gflags, runner_util,
# evalue_util, custom_ops and the quantized ops (no fp32 model needs them).
set(ET_WHOLE_ARCHIVES
  "${B}/libexecutorch.a"
  "${B}/backends/xnnpack/libxnnpack_backend.a"
  "${B}/configurations/liboptimized_native_cpu_ops_lib.a")
set(ET_ARCHIVES
  "${B}/extension/module/libextension_module_static.a"
  "${B}/extension/data_loader/libextension_data_loader.a"
  "${B}/extension/flat_tensor/libextension_flat_tensor.a"
  "${B}/extension/named_data_map/libextension_named_data_map.a"
  "${B}/extension/tensor/libextension_tensor.a"
  "${B}/libexecutorch.a"
  "${B}/configurations/liboptimized_native_cpu_ops_lib.a"
  "${B}/kernels/optimized/liboptimized_kernels.a"
  "${B}/kernels/portable/liboptimized_portable_kernels.a"
  "${B}/kernels/portable/cpu/util/libkernels_util_all_deps.a"
  "${B}/backends/xnnpack/libxnnpack_backend.a"
  "${B}/backends/xnnpack/third-party/XNNPACK/libXNNPACK.a"
  "${B}/backends/xnnpack/third-party/XNNPACK/libxnnpack-microkernels-prod.a"
  "${B}/kernels/optimized/libcpublas.a"
  "${B}/kernels/optimized/libeigen_blas.a"
  "${B}/extension/threadpool/libextension_threadpool.a"
  "${B}/libexecutorch_core.a"
  "${B}/backends/xnnpack/third-party/pthreadpool/libpthreadpool.a"
  "${B}/backends/xnnpack/third-party/cpuinfo/libcpuinfo.a")
foreach(a ${ET_WHOLE_ARCHIVES} ${ET_ARCHIVES})
  if(NOT EXISTS "${a}")
    message(FATAL_ERROR "missing ExecuTorch archive ${a}")
  endif()
endforeach()
set(_whole "")
foreach(a ${ET_WHOLE_ARCHIVES})
  list(APPEND _whole "-Wl,--whole-archive" "${a}" "-Wl,--no-whole-archive")
endforeach()

add_executable(et-server src/main.cpp)
# The patched c10 Macros.h picks musl's __assert_fail signature from
# EXECUTORCH_LIBC_MUSL; every translation unit that reaches it through module.h
# needs it. The ET_* values match the ExecuTorch tree's own compile lines, so
# inline header code agrees with the archives it links against.
target_compile_definitions(et-server PRIVATE
  EXECUTORCH_LIBC_MUSL _GNU_SOURCE C10_USING_CUSTOM_GENERATED_MACROS
  ET_ENABLE_DEPRECATED_CONSTANT_BUFFER=0 ET_ENABLE_PROGRAM_VERIFICATION=0
  ET_LOG_ENABLED=0 ET_MIN_LOG_LEVEL=Info)
target_compile_options(et-server PRIVATE -ffunction-sections -fdata-sections -Wno-deprecated-declarations)
target_include_directories(et-server PRIVATE
  "${EXECUTORCH_SRC}/.."
  "${EXECUTORCH_SRC}/runtime/core/portable_type/c10"
  "${EXECUTORCH_BUILD}/include"
  "${EXECUTORCH_SRC}/backends/xnnpack/third-party/pthreadpool/include"
  "${TOKENIZERS_CPP_SRC}/include")
target_link_options(et-server PRIVATE -Wl,--gc-sections)
target_link_libraries(et-server PRIVATE
  httplib::httplib nlohmann_json::nlohmann_json
  ${_whole}
  -Wl,--start-group ${ET_ARCHIVES} -Wl,--end-group
  "${TOKENIZERS_C_LIB}"
  ${CMAKE_DL_LIBS} pthread m)
