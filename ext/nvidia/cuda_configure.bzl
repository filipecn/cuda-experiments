"""Build rule generator for locally installed CUDA toolkit."""

def _get_env_var(repository_ctx, name, default):
    if name in repository_ctx.os.environ:
        return repository_ctx.os.environ[name]
    return default

def _impl(repository_ctx):
    cuda_path = _get_env_var(repository_ctx, "CUDA_PATH", "/opt/cuda")

    print("Using CUDA from %s\n" % cuda_path)

    repository_ctx.symlink(cuda_path, "cuda")

    repository_ctx.file("nvcc.sh", """
#! /bin/bash
repo_path=%s
compiler=${CC:+"--compiler-bindir=$CC"}
$repo_path/cuda/bin/nvcc $compiler --compiler-options=-fPIC --include-path=$repo_path $*
""" % repository_ctx.path("."))

    repository_ctx.file("BUILD", """
package(default_visibility = ["//visibility:public"])

config_setting(
    name = "linux",
    values = { "cpu": "k8" },
)
config_setting(
    name = "osx",   
    values = { "cpu": "darwin" },
)
config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"}
)

sh_binary(
    name = "nvcc",
    srcs = ["nvcc.sh"],
)
# The *_headers cc_library rules below aren't cc_inc_library rules because
# dependent targets would only see the first one.
cc_library(
    name = "cuda_headers",
    hdrs = select({"windows":glob(["cuda/include/**/*.h"]),
    "linux": glob(["cuda/include/**/*.h"]),
        "//conditions:default": []}
    ),
    # Allows including CUDA headers with angle brackets.
    includes = select({"windows":glob(["cuda/include"]),
    "linux": glob(["cuda/include"]),
        "//conditions:default": []}
    ),
)
cc_library(
    name = "cuda",
    srcs = select({"windows":["cuda/lib/x64/cuda.lib"],
    "linux": ["cuda/lib64/stubs/libcuda.so"],
        "//conditions:default": []}
    ),
    linkopts = ["-ldl"],
)
cc_library(
    name = "cuda_runtime",
    srcs = select({"windows":["cuda/lib/x64/cudart_static.lib"],
    "linux": ["cuda/lib64/libcudart_static.a"],
        "//conditions:default": []}
    ),
    deps = [":cuda"],
    linkopts = ["-lrt"],
)
cc_library(
    name = "cupti_headers",
    hdrs = select({"windows":glob(["cuda/extras/CUPTI/include/**/*.h"]),
    "linux": glob(["cuda/extras/CUPTI/include/**/*.h"]),
        "//conditions:default": []}
    ),
    # Allows including CUPTI headers with angle brackets.
    includes = select({"windows": ["cuda/extras/CUPTI/include"],
    "linux": ["cuda/extras/CUPTI/include"],
        "//conditions:default": []}
    ),
)
cc_library(
    name = "cupti",
    srcs = select({"windows":glob(["cuda/extras/CUPTI/lib64/cupti.lib"]),
    "linux": glob(["cuda/extras/CUPTI/lib64/libcupti.so*"]),
        "//conditions:default": []}
    ),
)
cc_library(
    name = "cuda_util",
    deps = [":cuda_util_compile"],
)
""")

cuda_configure = repository_rule(
    implementation = _impl,
    environ = ["CUDA_PATH"],
)