cc_binary(
    name = "hello-cuda",
    srcs = ["hello_cuda_cu"],
    deps = ["@ponosZIP//ponos-master/ponos:ponos",
        "@local_config_cuda//:cuda_headers",
        "@local_config_cuda//:cuda_runtime",
        "@local_config_cuda//:cupti",
        "@local_config_cuda//:cupti_headers",],
    linkopts = ['-lpthread']
)

genrule(
    name = "hello_cuda_cu",
    srcs = ["hello_cuda.cu"],
    outs = ["hello_cuda.o"],
    cmd = "$(location @local_config_cuda//:nvcc) --compile $(SRCS) --output-file $(OUTS)",
    tools = ["@local_config_cuda//:nvcc"],
)

