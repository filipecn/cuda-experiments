/// Copyright (c) 2020, FilipeCN.
///
/// The MIT License (MIT)
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.
///
///\file context.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-11
///
///\brief

#include "optix_context.h"
#include <iostream>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <hermes/hermes.h>

namespace optix {

static void context_log_cb(unsigned int level, const char *tag,
                           const char *message, void *) {
  fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

Context::Context() {
  // init optix
  cudaFree(0);
  int numDevices;
  cudaGetDeviceCount(&numDevices);
  if (numDevices == 0)
    throw std::runtime_error("#osc: no CUDA capable devices found!");
  std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;
  CHECK_OPTIX(optixInit());

  // retrieve cuda context
  const int device_id = 0;
  using namespace hermes::cuda;
  CUDA_CHECK(cudaSetDevice(device_id));
  CUDA_CHECK(cudaStreamCreate(&stream_));

  cudaGetDeviceProperties(&device_properties_, device_id);
  std::cout << "#osc: running on device: " << device_properties_.name
            << std::endl;

  CUresult cuRes = cuCtxGetCurrent(&cuda_context_);
  if (cuRes != CUDA_SUCCESS)
    fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

  // create optix context
  CHECK_OPTIX(optixDeviceContextCreate(cuda_context_, 0, &optix_context_));
  CHECK_OPTIX(optixDeviceContextSetLogCallback(optix_context_, context_log_cb,
                                               nullptr, 4));
}

OptixDeviceContext Context::handle() const { return optix_context_; }

CUstream Context::stream() const { return stream_; }

} // namespace optix