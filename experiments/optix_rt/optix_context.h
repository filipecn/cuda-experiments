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
///\file device.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-11
///
///\brief

#ifndef OPTIX_CONTEXT_H
#define OPTIX_CONTEXT_H

#include "debug.h"
#include <cuda_runtime.h>

namespace optix {

/// A context is used to manage a single GPU. The optix device context is
/// created by specifying the CUDA context associated with the device.
class Context {
public:
  Context();

  [[nodiscard]] OptixDeviceContext handle() const;
  [[nodiscard]] CUstream stream() const;

private:
  CUcontext cuda_context_;
  CUstream stream_;
  cudaDeviceProp device_properties_;

  OptixDeviceContext optix_context_;
};

} // namespace optix

#endif