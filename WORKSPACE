load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load ("//ext/nvidia:cuda_configure.bzl", "cuda_configure")

new_http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.7.0.zip",
    sha256 = "b58cb7547a28b2c718d1e38aee18a3659c9e3ff52440297e965f5edffe34b6d0",
    build_file = "ext/BUILD.gtest",
    strip_prefix = "googletest-release-1.7.0",
)

http_archive(
    name = "ponosZIP",
    url = "https://github.com/filipecn/ponos/archive/master.zip",
)

# ===== cuda =====
cuda_configure(name = "local_config_cuda")